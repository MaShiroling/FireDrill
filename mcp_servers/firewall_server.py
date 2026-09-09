"""假防火墙 MCP Server（有状态）

本地实现的防火墙设备模拟服务，提供：
- 安全域 / ACL 规则配置查询（读配置）
- 候选配置编辑 + commit/discard 两阶段下发（下发）
- 配置 diff、模拟报文转发验证（验证）
- 参数校验错误与可注入的设备故障（出错处理评测）

管理通道（/admin/* HTTP 路由，不注册为 MCP 工具）供自动化评测使用：
- POST /admin/reset    恢复出厂状态
- POST /admin/scenario 注入故障场景
- GET  /admin/snapshot 导出完整状态（含操作审计日志，供打分）
- GET  /admin/health   健康检查
"""

import copy
import functools
import ipaddress
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

try:
    from .firewall_repository import (
        ChangeSetConflict,
        FirewallRepository,
        RevisionConflict,
        SQLiteFirewallRepository,
    )
except ImportError:  # 兼容 ``python mcp_servers/firewall_server.py`` 和现有测试导入方式
    from firewall_repository import (
        ChangeSetConflict,
        FirewallRepository,
        RevisionConflict,
        SQLiteFirewallRepository,
    )

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Firewall_MCP_Server")

mcp = FastMCP("Firewall")


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        if kwargs:
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict) and result.get("success") is False:
                logger.info(f"返回状态: FAILED - {result.get('error')}")
            else:
                logger.info(f"返回状态: SUCCESS")
            logger.info(f"=" * 80)
            return result

        except Exception as e:
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


# ============================================================
# 防火墙状态（与 MCP 工具解耦，便于单元测试）
# ============================================================

# 故障注入模式
FAULT_MODES = ("none", "commit_reject", "commit_flaky", "commit_lose")

_PROTOCOLS = ("tcp", "udp", "icmp", "any")
_ACTIONS = ("allow", "deny")
_PORT_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class FirewallState:
    """防火墙设备状态。

    两阶段配置语义：
    - running_rules：已生效配置，模拟报文转发（test_traffic）以它为准
    - candidate_rules：候选配置，所有写操作（add/update/delete/move）都落在它上面
    - commit_config() 校验通过后候选配置才生效；discard_candidate() 放弃改动
    """

    def __init__(self, repository: Optional[FirewallRepository] = None) -> None:
        self._repository = repository
        persisted = repository.load() if repository else None
        if persisted:
            self._restore(persisted)
        else:
            self.reset()

    # ---------- 出厂配置 ----------

    def reset(self) -> None:
        """恢复出厂状态（清空故障注入、审计日志、命中计数）。"""
        self.hostname = "fake-fw-01"
        self.model = "MockWall-3000"
        self.firmware = "v5.2.1"
        self.zones: Dict[str, Dict[str, Any]] = {
            "trust": {"name": "trust", "subnet": "10.1.0.0/16", "description": "内网办公区"},
            "dmz": {"name": "dmz", "subnet": "172.16.1.0/24", "description": "对外服务区"},
            "untrust": {"name": "untrust", "subnet": "0.0.0.0/0", "description": "互联网"},
        }
        self._rule_seq = 0

        def mk(name, src_zone, dst_zone, src_addr, dst_addr, protocol, dst_port, action, desc):
            return self._new_rule(name, src_zone, dst_zone, src_addr, dst_addr,
                                  protocol, dst_port, action, desc)

        self.running_rules: List[Dict[str, Any]] = [
            mk("allow-web-http", "trust", "untrust", "any", "any", "tcp", "80", "allow",
               "内网访问互联网 HTTP"),
            mk("allow-web-https", "trust", "untrust", "any", "any", "tcp", "443", "allow",
               "内网访问互联网 HTTPS"),
            mk("allow-dns", "trust", "untrust", "any", "any", "udp", "53", "allow",
               "内网 DNS 解析"),
            mk("allow-public-to-web", "untrust", "dmz", "any", "172.16.1.10/32", "tcp", "443",
               "allow", "互联网访问 DMZ Web 服务"),
            mk("block-dmz-to-trust", "dmz", "trust", "any", "any", "any", "any", "deny",
               "禁止 DMZ 主动访问内网"),
            mk("default-deny", "any", "any", "any", "any", "any", "any", "deny",
               "默认拒绝所有流量"),
        ]
        self.running_revision = 1
        self.candidate_rules: List[Dict[str, Any]] = copy.deepcopy(self.running_rules)
        self.hit_counts: Dict[str, int] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.fault: Dict[str, Any] = {"mode": "none", "fail_times": 0}
        self.candidate_base_revision = self.running_revision
        self.change_set_id = "memory"
        if self._repository:
            metadata = self._repository.reset(self._storage_state())
            self.change_set_id = metadata["change_set_id"]
            self.candidate_base_revision = metadata["candidate_base_revision"]

    def _restore(self, state: Dict[str, Any]) -> None:
        """从仓储快照恢复进程内状态。"""
        self.hostname = state["hostname"]
        self.model = state["model"]
        self.firmware = state["firmware"]
        self.zones = copy.deepcopy(state["zones"])
        self.running_revision = state["running_revision"]
        self.running_rules = copy.deepcopy(state["running_rules"])
        self.candidate_rules = copy.deepcopy(state["candidate_rules"])
        self.candidate_base_revision = state["candidate_base_revision"]
        self.change_set_id = state["change_set_id"]
        self._rule_seq = state["rule_seq"]
        self.hit_counts = copy.deepcopy(state["hit_counts"])
        self.fault = copy.deepcopy(state["fault"])
        self.audit_log = copy.deepcopy(state["audit_log"])

    def _storage_state(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "model": self.model,
            "firmware": self.firmware,
            "zones": self.zones,
            "running_revision": self.running_revision,
            "running_rules": self.running_rules,
            "candidate_rules": self.candidate_rules,
            "candidate_base_revision": self.candidate_base_revision,
            "change_set_id": self.change_set_id,
            "rule_seq": self._rule_seq,
            "hit_counts": self.hit_counts,
            "fault": self.fault,
            "now": _now(),
        }

    # ---------- 审计 ----------

    def _audit(
        self,
        operation: str,
        params: Dict[str, Any],
        result: str,
        detail: str,
        *,
        persist: bool = True,
    ) -> Dict[str, Any]:
        item = {
            "timestamp": _now(),
            "operation": operation,
            "params": params,
            "result": result,
            "detail": detail,
            "running_revision": self.running_revision,
        }
        self.audit_log.append(item)
        if persist and self._repository:
            self._repository.save_working(self._storage_state(), item)
        return item

    def set_fault(self, mode: str, fail_times: int = 0) -> None:
        """设置并持久化故障注入状态（管理面使用）。"""
        self.fault = {"mode": mode, "fail_times": fail_times}
        self._audit("set_fault", {"mode": mode, "fail_times": fail_times},
                    "success", f"故障注入已设置为 {mode}")

    # ---------- 校验 ----------

    def _validate_zone(self, zone: str, field: str) -> Optional[str]:
        if zone != "any" and zone not in self.zones:
            return f"{field} 非法: '{zone}'，可选值为 {sorted(self.zones)} 或 'any'"
        return None

    @staticmethod
    def _validate_addr(addr: str, field: str) -> Optional[str]:
        if addr == "any":
            return None
        try:
            ipaddress.ip_network(addr, strict=False)
        except ValueError:
            return f"{field} 非法: '{addr}'，需为 CIDR（如 10.1.0.0/16）或 'any'"
        return None

    @staticmethod
    def _validate_port(port: str) -> Optional[str]:
        if port == "any":
            return None
        m = _PORT_RANGE_RE.match(port)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 1 <= lo <= hi <= 65535:
                return None
            return f"dst_port 非法: '{port}'，端口范围需在 1-65535 且起始不大于结束"
        if port.isdigit() and 1 <= int(port) <= 65535:
            return None
        return f"dst_port 非法: '{port}'，需为 1-65535 的端口、a-b 范围或 'any'"

    def _validate_rule_fields(self, fields: Dict[str, Any]) -> Optional[str]:
        """校验规则字段集合，返回错误信息或 None。"""
        for f in ("src_zone", "dst_zone"):
            if f in fields:
                err = self._validate_zone(fields[f], f)
                if err:
                    return err
        for f in ("src_addr", "dst_addr"):
            if f in fields:
                err = self._validate_addr(fields[f], f)
                if err:
                    return err
        if "protocol" in fields and fields["protocol"] not in _PROTOCOLS:
            return f"protocol 非法: '{fields['protocol']}'，可选值为 {_PROTOCOLS}"
        if "action" in fields and fields["action"] not in _ACTIONS:
            return f"action 非法: '{fields['action']}'，可选值为 {_ACTIONS}"
        if "dst_port" in fields:
            err = self._validate_port(str(fields["dst_port"]))
            if err:
                return err
        return None

    @staticmethod
    def _five_tuple(rule: Dict[str, Any]) -> tuple:
        return (rule["src_zone"], rule["dst_zone"], rule["src_addr"], rule["dst_addr"],
                rule["protocol"], rule["dst_port"], rule["action"])

    def _find_duplicate(self, candidate: Dict[str, Any], exclude_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        for r in self.candidate_rules:
            if r["rule_id"] != exclude_id and self._five_tuple(r) == self._five_tuple(candidate):
                return r
        return None

    def _find_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        for r in self.candidate_rules:
            if r["rule_id"] == rule_id:
                return r
        return None

    def _new_rule(self, name, src_zone, dst_zone, src_addr, dst_addr,
                  protocol, dst_port, action, description) -> Dict[str, Any]:
        self._rule_seq += 1
        return {
            "rule_id": f"rule-{self._rule_seq:03d}",
            "name": name,
            "src_zone": src_zone,
            "dst_zone": dst_zone,
            "src_addr": src_addr,
            "dst_addr": dst_addr,
            "protocol": protocol,
            "dst_port": str(dst_port),
            "action": action,
            "enabled": True,
            "description": description,
        }

    # ---------- 写操作（作用于候选配置） ----------

    def add_rule(self, name, src_zone, dst_zone, src_addr, dst_addr,
                 protocol, dst_port, action, description) -> Dict[str, Any]:
        params = locals().copy()
        params.pop("self")
        rule = self._new_rule(name, src_zone, dst_zone, src_addr, dst_addr,
                              protocol, dst_port, action, description)
        err = self._validate_rule_fields(rule)
        if err:
            self._audit("add_rule", params, "error", err)
            return {"success": False, "error": err}
        dup = self._find_duplicate(rule)
        if dup:
            err = (f"规则与现有候选规则重复: {dup['rule_id']} ({dup['name']})，"
                   "五元组与动作完全一致")
            self._audit("add_rule", params, "error", err)
            return {"success": False, "error": err}
        # 插到 default-deny 之前（若存在），保持兜底规则在最后
        idx = next((i for i, r in enumerate(self.candidate_rules) if r["name"] == "default-deny"),
                   len(self.candidate_rules))
        self.candidate_rules.insert(idx, rule)
        self._audit("add_rule", params, "success", f"新增候选规则 {rule['rule_id']}")
        return {"success": True, "rule": rule, "position": idx + 1,
                "pending_changes": True,
                "message": "规则已加入候选配置，需 commit_config 后生效"}

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        params = {"rule_id": rule_id, **updates}
        rule = self._find_rule(rule_id)
        if not rule:
            err = f"规则不存在: {rule_id}"
            self._audit("update_rule", params, "error", err)
            return {"success": False, "error": err}
        merged = {**rule, **{k: v for k, v in updates.items() if v is not None}}
        if "dst_port" in merged:
            merged["dst_port"] = str(merged["dst_port"])
        err = self._validate_rule_fields(merged)
        if err:
            self._audit("update_rule", params, "error", err)
            return {"success": False, "error": err}
        dup = self._find_duplicate(merged, exclude_id=rule_id)
        if dup:
            err = f"修改后与候选规则重复: {dup['rule_id']} ({dup['name']})"
            self._audit("update_rule", params, "error", err)
            return {"success": False, "error": err}
        rule.update({k: v for k, v in merged.items() if k != "rule_id"})
        self._audit("update_rule", params, "success", f"更新候选规则 {rule_id}")
        return {"success": True, "rule": rule, "pending_changes": True,
                "message": "候选规则已更新，需 commit_config 后生效"}

    def delete_rule(self, rule_id: str) -> Dict[str, Any]:
        rule = self._find_rule(rule_id)
        if not rule:
            err = f"规则不存在: {rule_id}"
            self._audit("delete_rule", {"rule_id": rule_id}, "error", err)
            return {"success": False, "error": err}
        self.candidate_rules.remove(rule)
        self._audit("delete_rule", {"rule_id": rule_id}, "success",
                    f"删除候选规则 {rule_id} ({rule['name']})")
        return {"success": True, "deleted_rule": rule, "pending_changes": True,
                "message": "候选规则已删除，需 commit_config 后生效"}

    def move_rule(self, rule_id: str, position: int) -> Dict[str, Any]:
        params = {"rule_id": rule_id, "position": position}
        rule = self._find_rule(rule_id)
        if not rule:
            err = f"规则不存在: {rule_id}"
            self._audit("move_rule", params, "error", err)
            return {"success": False, "error": err}
        if position < 1 or position > len(self.candidate_rules):
            err = f"position 非法: {position}，有效范围 1-{len(self.candidate_rules)}"
            self._audit("move_rule", params, "error", err)
            return {"success": False, "error": err}
        self.candidate_rules.remove(rule)
        self.candidate_rules.insert(position - 1, rule)
        self._audit("move_rule", params, "success",
                    f"规则 {rule_id} 移动到第 {position} 位")
        return {"success": True, "rule_id": rule_id, "position": position,
                "pending_changes": True,
                "message": "候选规则顺序已调整，需 commit_config 后生效"}

    # ---------- 提交 / 回滚 ----------

    def _rules_equal(self) -> bool:
        return json.dumps(self.candidate_rules, sort_keys=True) == \
               json.dumps(self.running_rules, sort_keys=True)

    def commit(
        self,
        idempotency_key: Optional[str] = None,
        expected_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        params = {k: v for k, v in {
            "idempotency_key": idempotency_key,
            "expected_revision": expected_revision,
        }.items() if v is not None}
        if self._repository and idempotency_key:
            previous = self._repository.get_commit_result(idempotency_key)
            if previous is not None:
                persisted = self._repository.load()
                if persisted:
                    self._restore(persisted)
                return previous

        if self._rules_equal():
            err = "候选配置与运行配置一致，无改动可提交"
            self._audit("commit", params, "error", err)
            return {"success": False, "error": err}

        base_revision = (self.candidate_base_revision
                         if expected_revision is None else expected_revision)
        if base_revision != self.candidate_base_revision:
            err = (f"版本冲突: 请求期望 R{base_revision}，"
                   f"当前 Candidate 基于 R{self.candidate_base_revision}")
            self._audit("commit", params, "error", err)
            return {"success": False, "error": err, "conflict": True}

        mode = self.fault.get("mode", "none")
        if mode == "commit_reject":
            err = "设备错误: 配置提交被设备拒绝（fault: commit_reject）"
            self._audit("commit", params, "error", err)
            return {"success": False, "error": err}
        if mode == "commit_flaky" and self.fault.get("fail_times", 0) > 0:
            self.fault["fail_times"] -= 1
            err = "设备繁忙: 配置提交超时，请稍后重试（fault: commit_flaky）"
            self._audit("commit", params, "error", err)
            return {"success": False, "error": err}

        # 生效
        self.running_rules = copy.deepcopy(self.candidate_rules)
        self.running_revision += 1
        if mode == "commit_lose":
            # 配置实际已生效，但设备未回 ACK —— 歧义故障
            err = "提交超时: 设备未返回确认，配置状态未知（fault: commit_lose）"
            response = {
                "success": False,
                "error": err,
                "hint": "可通过 get_firewall_overview / get_config_diff 核实设备实际状态",
            }
            audit = self._audit(
                "commit", params, "error",
                f"{err}；实际已生效，running_revision={self.running_revision}",
                persist=False,
            )
        else:
            response = {
                "success": True,
                "running_revision": self.running_revision,
                "message": f"配置提交成功，当前运行版本 R{self.running_revision}",
            }
            audit = self._audit(
                "commit", params, "success",
                f"候选配置已生效，running_revision={self.running_revision}",
                persist=False,
            )

        if self._repository:
            commit_key = idempotency_key or f"changeset:{self.change_set_id}:commit"
            try:
                self.change_set_id = self._repository.commit(
                    self._storage_state(), base_revision, commit_key, response, audit,
                )
                self.candidate_base_revision = self.running_revision
            except (RevisionConflict, ChangeSetConflict) as exc:
                persisted = self._repository.load()
                if persisted:
                    self._restore(persisted)
                self._audit("commit", params, "error", str(exc))
                return {"success": False, "error": str(exc), "conflict": True}
        else:
            self.candidate_base_revision = self.running_revision
        return response

    def discard(self) -> Dict[str, Any]:
        if self._rules_equal():
            err = "候选配置无改动，无需放弃"
            self._audit("discard", {}, "error", err)
            return {"success": False, "error": err}
        self.candidate_rules = copy.deepcopy(self.running_rules)
        audit = self._audit("discard", {}, "success", "候选配置已回滚到运行配置",
                            persist=False)
        if self._repository:
            try:
                self.change_set_id = self._repository.discard(self._storage_state(), audit)
                self.candidate_base_revision = self.running_revision
            except (RevisionConflict, ChangeSetConflict) as exc:
                persisted = self._repository.load()
                if persisted:
                    self._restore(persisted)
                self._audit("discard", {}, "error", str(exc))
                return {"success": False, "error": str(exc), "conflict": True}
        else:
            self.candidate_base_revision = self.running_revision
        return {"success": True, "message": "候选配置已放弃，恢复为当前运行配置"}

    # ---------- 验证 ----------

    def diff(self) -> Dict[str, Any]:
        running_by_id = {r["rule_id"]: r for r in self.running_rules}
        candidate_by_id = {r["rule_id"]: r for r in self.candidate_rules}

        added = [r for rid, r in candidate_by_id.items() if rid not in running_by_id]
        removed = [r for rid, r in running_by_id.items() if rid not in candidate_by_id]
        modified, moved = [], []
        for idx, r in enumerate(self.candidate_rules):
            rid = r["rule_id"]
            if rid in running_by_id:
                if r != running_by_id[rid]:
                    modified.append({"rule_id": rid,
                                     "candidate": r, "running": running_by_id[rid]})
                old_idx = next(i for i, x in enumerate(self.running_rules) if x["rule_id"] == rid)
                if old_idx != idx and r == running_by_id[rid]:
                    moved.append({"rule_id": rid, "from": old_idx + 1, "to": idx + 1})

        return {
            "success": True,
            "has_changes": bool(added or removed or modified or moved),
            "added": added, "removed": removed, "modified": modified, "moved": moved,
            "running_revision": self.running_revision,
            "candidate_base_revision": self.candidate_base_revision,
            "change_set_id": self.change_set_id,
        }

    @staticmethod
    def _addr_match(rule_addr: str, packet_addr: str) -> bool:
        if rule_addr == "any":
            return True
        try:
            return ipaddress.ip_address(packet_addr) in ipaddress.ip_network(rule_addr, strict=False)
        except ValueError:
            return False

    @staticmethod
    def _port_match(rule_port: str, packet_port: str) -> bool:
        if rule_port == "any":
            return True
        m = _PORT_RANGE_RE.match(rule_port)
        if m:
            return int(m.group(1)) <= int(packet_port) <= int(m.group(2))
        return rule_port == str(packet_port)

    def test_traffic(self, src_zone, dst_zone, src_addr, dst_addr,
                     protocol, dst_port) -> Dict[str, Any]:
        params = {"src_zone": src_zone, "dst_zone": dst_zone, "src_addr": src_addr,
                  "dst_addr": dst_addr, "protocol": protocol, "dst_port": dst_port}
        for err in (self._validate_zone(src_zone, "src_zone"),
                    self._validate_zone(dst_zone, "dst_zone")):
            if err:
                self._audit("test_traffic", params, "error", err)
                return {"success": False, "error": err}
        if protocol not in _PROTOCOLS or protocol == "any":
            err = f"protocol 非法: '{protocol}'，模拟报文需为 tcp/udp/icmp"
            self._audit("test_traffic", params, "error", err)
            return {"success": False, "error": err}
        if protocol != "icmp":
            err = self._validate_port(str(dst_port))
            if err or str(dst_port) == "any":
                err = err or "模拟报文的 dst_port 不能为 'any'"
                self._audit("test_traffic", params, "error", err)
                return {"success": False, "error": err}

        for idx, rule in enumerate(self.running_rules):
            if not rule["enabled"]:
                continue
            if rule["src_zone"] not in ("any", src_zone):
                continue
            if rule["dst_zone"] not in ("any", dst_zone):
                continue
            if not self._addr_match(rule["src_addr"], src_addr):
                continue
            if not self._addr_match(rule["dst_addr"], dst_addr):
                continue
            if rule["protocol"] not in ("any", protocol):
                continue
            if protocol != "icmp" and not self._port_match(rule["dst_port"], str(dst_port)):
                continue
            self.hit_counts[rule["rule_id"]] = self.hit_counts.get(rule["rule_id"], 0) + 1
            self._audit("test_traffic", params, "success",
                        f"命中 {rule['rule_id']} -> {rule['action']}")
            return {
                "success": True, "matched": True, "position": idx + 1,
                "matched_rule": rule, "action": rule["action"],
                "message": f"报文命中第 {idx + 1} 条规则 {rule['rule_id']} "
                           f"({rule['name']})，动作: {rule['action']}",
            }
        self._audit("test_traffic", params, "success", "未命中任何规则 -> implicit-deny")
        return {"success": True, "matched": False, "action": "deny",
                "message": "报文未命中任何规则，按隐式拒绝处理"}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname, "model": self.model, "firmware": self.firmware,
            "zones": self.zones,
            "running_revision": self.running_revision,
            "candidate_base_revision": self.candidate_base_revision,
            "change_set_id": self.change_set_id,
            "running_rules": self.running_rules,
            "candidate_rules": self.candidate_rules,
            "pending_changes": not self._rules_equal(),
            "hit_counts": self.hit_counts,
            "fault": self.fault,
            "audit_log": self.audit_log,
            "persistence": "sqlite" if self._repository else "memory",
        }


def _create_global_state() -> FirewallState:
    backend = os.getenv("FIREWALL_STATE_BACKEND", "sqlite").lower()
    if backend == "memory":
        logger.info("防火墙状态后端: memory")
        return FirewallState()
    db_path = os.getenv(
        "FIREWALL_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "volumes" / "firewall_state.db"),
    )
    try:
        repository = SQLiteFirewallRepository(db_path)
        logger.info("防火墙状态后端: sqlite (%s)", db_path)
        return FirewallState(repository=repository)
    except Exception:
        logger.exception("SQLite 状态初始化失败，降级为 memory")
        return FirewallState()


# 全局单例状态；服务默认持久化，测试可直接实例化 FirewallState() 使用纯内存。
STATE = _create_global_state()


# ============================================================
# Agent 侧 MCP 工具：读配置
# ============================================================

@mcp.tool()
@log_tool_call
def get_firewall_overview() -> Dict[str, Any]:
    """获取防火墙设备概览：设备信息、运行/候选版本、是否有未提交改动。

    Returns:
        Dict: hostname、model、firmware、running_revision、pending_changes 等
    """
    STATE._audit("get_firewall_overview", {}, "success", "查询设备概览")
    return {
        "success": True,
        "hostname": STATE.hostname,
        "model": STATE.model,
        "firmware": STATE.firmware,
        "running_revision": STATE.running_revision,
        "candidate_base_revision": STATE.candidate_base_revision,
        "change_set_id": STATE.change_set_id,
        "rule_count": len(STATE.running_rules),
        "pending_changes": not STATE._rules_equal(),
    }


@mcp.tool()
@log_tool_call
def list_security_zones() -> Dict[str, Any]:
    """列出防火墙的安全域（trust/dmz/untrust）及各自网段。

    Returns:
        Dict: zones 列表
    """
    return {"success": True, "zones": list(STATE.zones.values())}


@mcp.tool()
@log_tool_call
def list_firewall_rules(source: str = "running") -> Dict[str, Any]:
    """按匹配顺序列出防火墙 ACL 规则（top-down，首条命中生效）。

    Args:
        source: 配置来源，"running"（已生效，默认）或 "candidate"（候选未提交）

    Returns:
        Dict: rules 列表，含 rule_id、name、五元组、action、enabled、顺序 position
    """
    if source not in ("running", "candidate"):
        return {"success": False, "error": f"source 非法: '{source}'，可选 running/candidate"}
    rules = STATE.running_rules if source == "running" else STATE.candidate_rules
    STATE._audit("list_firewall_rules", {"source": source}, "success",
                 f"查询规则列表（{source}，{len(rules)} 条）")
    return {
        "success": True,
        "source": source,
        "running_revision": STATE.running_revision,
        "rules": [{"position": i + 1, **r} for i, r in enumerate(rules)],
    }


@mcp.tool()
@log_tool_call
def get_firewall_rule(rule_id: str) -> Dict[str, Any]:
    """查看单条防火墙规则详情（优先查候选配置，其次运行配置）。

    Args:
        rule_id: 规则 ID，如 "rule-001"

    Returns:
        Dict: 规则详情及所在配置（candidate/running）
    """
    for source, rules in (("candidate", STATE.candidate_rules),
                          ("running", STATE.running_rules)):
        for r in rules:
            if r["rule_id"] == rule_id:
                STATE._audit("get_firewall_rule", {"rule_id": rule_id}, "success",
                             f"查询规则详情（{source}）")
                return {"success": True, "source": source, "rule": r}
    return {"success": False, "error": f"规则不存在: {rule_id}"}


# ============================================================
# Agent 侧 MCP 工具：下发（作用于候选配置，commit 后生效）
# ============================================================

@mcp.tool()
@log_tool_call
def add_firewall_rule(
    name: str,
    src_zone: str,
    dst_zone: str,
    src_addr: str,
    dst_addr: str,
    protocol: str,
    dst_port: str,
    action: str,
    description: str = "",
) -> Dict[str, Any]:
    """向候选配置添加一条 ACL 规则（插入在 default-deny 之前，需 commit_config 生效）。

    Args:
        name: 规则名称，如 "allow-office-to-gitlab"
        src_zone: 源安全域，trust/dmz/untrust 或 any
        dst_zone: 目的安全域，trust/dmz/untrust 或 any
        src_addr: 源地址，CIDR（如 10.1.0.0/16）或 any
        dst_addr: 目的地址，CIDR 或 any
        protocol: 协议，tcp/udp/icmp/any
        dst_port: 目的端口，1-65535、a-b 范围或 any，如 "22" 或 "8000-9000"
        action: 动作，allow 或 deny
        description: 备注说明（可选）

    Returns:
        Dict: success 时含新规则（含自动分配的 rule_id）；失败时含 error 原因
              （非法参数、与现有规则重复等）
    """
    return STATE.add_rule(name, src_zone, dst_zone, src_addr, dst_addr,
                          protocol, str(dst_port), action, description)


@mcp.tool()
@log_tool_call
def update_firewall_rule(
    rule_id: str,
    name: Optional[str] = None,
    src_addr: Optional[str] = None,
    dst_addr: Optional[str] = None,
    protocol: Optional[str] = None,
    dst_port: Optional[str] = None,
    action: Optional[str] = None,
    enabled: Optional[bool] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """更新候选配置中已有规则的部分字段（需 commit_config 生效）。

    Args:
        rule_id: 要修改的规则 ID
        name/src_addr/dst_addr/protocol/dst_port/action/enabled/description:
            需要修改的字段，不传则不修改

    Returns:
        Dict: success 时含更新后的规则；失败时含 error 原因
    """
    updates = {k: v for k, v in {
        "name": name, "src_addr": src_addr, "dst_addr": dst_addr, "protocol": protocol,
        "dst_port": dst_port, "action": action, "enabled": enabled, "description": description,
    }.items() if v is not None}
    if not updates:
        return {"success": False, "error": "未提供任何待修改字段"}
    return STATE.update_rule(rule_id, updates)


@mcp.tool()
@log_tool_call
def delete_firewall_rule(rule_id: str) -> Dict[str, Any]:
    """从候选配置删除一条规则（需 commit_config 生效）。

    Args:
        rule_id: 要删除的规则 ID

    Returns:
        Dict: success 时含被删规则；失败时含 error 原因
    """
    return STATE.delete_rule(rule_id)


@mcp.tool()
@log_tool_call
def move_firewall_rule(rule_id: str, position: int) -> Dict[str, Any]:
    """调整候选规则顺序（规则按 position 自上而下首条命中，需 commit_config 生效）。

    Args:
        rule_id: 要移动的规则 ID
        position: 目标位置（从 1 开始）

    Returns:
        Dict: success 时含新位置；失败时含 error 原因
    """
    return STATE.move_rule(rule_id, position)


@mcp.tool()
@log_tool_call
def commit_config(
    idempotency_key: Optional[str] = None,
    expected_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """提交候选配置使其生效（两阶段下发的第二阶段）。

    提交前请先用 get_config_diff 确认改动符合预期。
    提交可能失败（设备繁忙/拒绝等），失败时候选配置保留，可重试或 discard_candidate 放弃。

    Args:
        idempotency_key: 可选幂等键；网络超时后使用同一个键重试不会重复提交
        expected_revision: 可选乐观锁版本，应等于 overview/diff 中的 running_revision

    Returns:
        Dict: success 时含新的 running_revision；失败时含 error 原因
    """
    return STATE.commit(idempotency_key=idempotency_key, expected_revision=expected_revision)


@mcp.tool()
@log_tool_call
def discard_candidate() -> Dict[str, Any]:
    """放弃候选配置的全部改动，回滚到当前运行配置。

    Returns:
        Dict: success 或 error（无改动可放弃时）
    """
    return STATE.discard()


# ============================================================
# Agent 侧 MCP 工具：验证
# ============================================================

@mcp.tool()
@log_tool_call
def get_config_diff() -> Dict[str, Any]:
    """对比候选配置与运行配置的差异。

    Returns:
        Dict: has_changes、added/removed/modified/moved 规则列表、running_revision
    """
    STATE._audit("get_config_diff", {}, "success", "查询配置差异")
    return STATE.diff()


@mcp.tool()
@log_tool_call
def test_traffic(
    src_zone: str,
    dst_zone: str,
    src_addr: str,
    dst_addr: str,
    protocol: str,
    dst_port: str = "80",
) -> Dict[str, Any]:
    """模拟一个报文穿越防火墙（按当前已生效 running 配置首条命中匹配），用于验证下发结果。

    Args:
        src_zone: 源安全域，trust/dmz/untrust
        dst_zone: 目的安全域，trust/dmz/untrust
        src_addr: 源 IP，如 "10.1.2.3"
        dst_addr: 目的 IP，如 "172.16.1.10"
        protocol: 协议，tcp/udp/icmp
        dst_port: 目的端口（icmp 时忽略），默认 "80"

    Returns:
        Dict: matched（是否命中规则）、matched_rule、action（allow/deny）；
              命中时累计该规则的 hit_count
    """
    return STATE.test_traffic(src_zone, dst_zone, src_addr, dst_addr, protocol, str(dst_port))


@mcp.tool()
@log_tool_call
def get_rule_hit_count(rule_id: Optional[str] = None) -> Dict[str, Any]:
    """查询规则的模拟报文命中计数（test_traffic 累计）。

    Args:
        rule_id: 规则 ID；不传则返回全部规则的计数

    Returns:
        Dict: hit_counts 映射
    """
    if rule_id:
        return {"success": True, "rule_id": rule_id,
                "hit_count": STATE.hit_counts.get(rule_id, 0)}
    return {"success": True, "hit_counts": dict(STATE.hit_counts)}


# ============================================================
# 评测管理通道（HTTP custom routes，不注册为 MCP 工具）
# ============================================================

@mcp.custom_route("/admin/health", methods=["GET"])
async def admin_health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "firewall",
                         "running_revision": STATE.running_revision})


@mcp.custom_route("/admin/reset", methods=["POST"])
async def admin_reset(request: Request) -> JSONResponse:
    """恢复出厂状态。body 可选 {"keep_fault": true} 保留故障注入。"""
    keep_fault = False
    try:
        body = await request.json()
        keep_fault = bool(body.get("keep_fault", False))
    except Exception:
        pass
    fault = copy.deepcopy(STATE.fault) if keep_fault else None
    STATE.reset()
    if fault:
        STATE.set_fault(fault["mode"], int(fault.get("fail_times", 0)))
    logger.info("[admin] 状态已重置" + ("（保留故障注入）" if fault else ""))
    return JSONResponse({"success": True, "message": "已恢复出厂状态",
                         "running_revision": STATE.running_revision})


@mcp.custom_route("/admin/scenario", methods=["POST"])
async def admin_scenario(request: Request) -> JSONResponse:
    """注入故障场景。body: {"fault": "none|commit_reject|commit_flaky|commit_lose",
    "fail_times": N}（fail_times 仅 commit_flaky 使用）。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "请求体需为 JSON"}, status_code=400)
    mode = body.get("fault")
    if mode not in FAULT_MODES:
        return JSONResponse(
            {"success": False, "error": f"fault 非法: {mode!r}，可选 {list(FAULT_MODES)}"},
            status_code=400)
    STATE.set_fault(mode, int(body.get("fail_times", 0)))
    logger.info(f"[admin] 故障注入已设置: {STATE.fault}")
    return JSONResponse({"success": True, "fault": STATE.fault})


@mcp.custom_route("/admin/snapshot", methods=["GET"])
async def admin_snapshot(request: Request) -> JSONResponse:
    """导出完整状态（运行/候选规则、revision、命中计数、故障注入、操作审计日志）。"""
    return JSONResponse(STATE.snapshot())


if __name__ == "__main__":
    # 使用 streamable-http 模式，运行在 8005 端口
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8005, path="/mcp")
