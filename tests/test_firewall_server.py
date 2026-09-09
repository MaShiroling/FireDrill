"""firewall_server.FirewallState 单元测试

不依赖 MCP 传输层，直接对状态类测试：
读配置 / 下发（候选 + commit/discard）/ 验证 / 故障注入。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_servers"))

from firewall_server import FirewallState


@pytest.fixture()
def fw() -> FirewallState:
    return FirewallState()


def _add_ok(fw: FirewallState, name: str = "allow-office-ssh", dst_port: str = "22") -> dict:
    return fw.add_rule(name, "trust", "dmz", "10.1.8.0/24", "172.16.1.20/32",
                       "tcp", dst_port, "allow", "测试规则")


# ---------- 出厂配置 / reset ----------

class TestFactory:
    def test_factory_defaults(self, fw):
        assert fw.running_revision == 1
        assert len(fw.running_rules) == 6
        assert set(fw.zones) == {"trust", "dmz", "untrust"}
        assert fw.running_rules[-1]["name"] == "default-deny"
        assert fw.candidate_rules == fw.running_rules
        assert fw.fault["mode"] == "none"

    def test_reset_restores(self, fw):
        _add_ok(fw)
        fw.commit()
        fw.fault = {"mode": "commit_reject", "fail_times": 0}
        fw.reset()
        assert fw.running_revision == 1
        assert len(fw.running_rules) == 6
        assert fw.fault["mode"] == "none"
        assert fw.audit_log == []


# ---------- 下发：add/update/delete/move ----------

class TestPush:
    def test_add_success_before_default_deny(self, fw):
        r = _add_ok(fw)
        assert r["success"] is True
        assert r["rule"]["rule_id"] == "rule-007"
        # 插入在 default-deny 之前
        assert fw.candidate_rules[-2]["rule_id"] == "rule-007"
        assert fw.candidate_rules[-1]["name"] == "default-deny"
        # 运行配置不受影响
        assert len(fw.running_rules) == 6

    def test_add_invalid_zone(self, fw):
        r = fw.add_rule("x", "office", "dmz", "any", "any", "tcp", "22", "allow", "")
        assert r["success"] is False
        assert "src_zone" in r["error"]

    def test_add_invalid_cidr(self, fw):
        r = fw.add_rule("x", "trust", "dmz", "999.1.1.1/32", "any", "tcp", "22", "allow", "")
        assert r["success"] is False
        assert "src_addr" in r["error"]

    def test_add_invalid_port(self, fw):
        r = fw.add_rule("x", "trust", "dmz", "any", "any", "tcp", "70000", "allow", "")
        assert r["success"] is False
        assert "dst_port" in r["error"]

    def test_add_duplicate_rejected(self, fw):
        assert _add_ok(fw)["success"] is True
        r = _add_ok(fw, name="dup")
        assert r["success"] is False
        assert "重复" in r["error"]

    def test_update_success(self, fw):
        rid = _add_ok(fw)["rule"]["rule_id"]
        r = fw.update_rule(rid, {"dst_port": "2222"})
        assert r["success"] is True
        assert fw._find_rule(rid)["dst_port"] == "2222"

    def test_update_not_found(self, fw):
        r = fw.update_rule("rule-999", {"dst_port": "22"})
        assert r["success"] is False
        assert "不存在" in r["error"]

    def test_update_invalid_field(self, fw):
        rid = _add_ok(fw)["rule"]["rule_id"]
        r = fw.update_rule(rid, {"protocol": "gre"})
        assert r["success"] is False
        assert "protocol" in r["error"]

    def test_delete_success(self, fw):
        rid = _add_ok(fw)["rule"]["rule_id"]
        r = fw.delete_rule(rid)
        assert r["success"] is True
        assert fw._find_rule(rid) is None

    def test_delete_not_found(self, fw):
        assert fw.delete_rule("rule-999")["success"] is False

    def test_move_rule(self, fw):
        rid = _add_ok(fw)["rule"]["rule_id"]
        r = fw.move_rule(rid, 1)
        assert r["success"] is True
        assert fw.candidate_rules[0]["rule_id"] == rid
        assert fw.move_rule(rid, 99)["success"] is False


# ---------- commit / discard ----------

class TestCommit:
    def test_commit_no_changes(self, fw):
        r = fw.commit()
        assert r["success"] is False
        assert "无改动" in r["error"]

    def test_commit_applies(self, fw):
        rid = _add_ok(fw)["rule"]["rule_id"]
        r = fw.commit()
        assert r["success"] is True
        assert fw.running_revision == 2
        assert any(x["rule_id"] == rid for x in fw.running_rules)
        # commit 后候选与运行一致
        assert fw.commit()["success"] is False

    def test_discard_reverts(self, fw):
        _add_ok(fw)
        r = fw.discard()
        assert r["success"] is True
        assert fw.candidate_rules == fw.running_rules
        assert fw.discard()["success"] is False  # 无改动可放弃

    def test_diff(self, fw):
        rid_add = _add_ok(fw)["rule"]["rule_id"]
        fw.update_rule("rule-001", {"dst_port": "8080"})
        fw.delete_rule("rule-002")
        fw.move_rule("rule-003", 1)
        d = fw.diff()
        assert d["has_changes"] is True
        assert [r["rule_id"] for r in d["added"]] == [rid_add]
        assert [r["rule_id"] for r in d["removed"]] == ["rule-002"]
        assert d["modified"][0]["rule_id"] == "rule-001"
        assert any(m["rule_id"] == "rule-003" for m in d["moved"])


# ---------- 验证：test_traffic / hit_count ----------

class TestVerify:
    def test_first_match_allow(self, fw):
        r = fw.test_traffic("trust", "untrust", "10.1.2.3", "8.8.8.8", "tcp", "443")
        assert r["matched"] is True
        assert r["matched_rule"]["rule_id"] == "rule-002"
        assert r["action"] == "allow"

    def test_hit_count_accumulates(self, fw):
        fw.test_traffic("trust", "untrust", "10.1.2.3", "8.8.8.8", "tcp", "443")
        fw.test_traffic("trust", "untrust", "10.1.2.4", "1.1.1.1", "tcp", "443")
        assert fw.hit_counts["rule-002"] == 2

    def test_deny_rule_match(self, fw):
        r = fw.test_traffic("dmz", "trust", "172.16.1.10", "10.1.5.5", "tcp", "22")
        assert r["matched"] is True
        assert r["action"] == "deny"
        assert r["matched_rule"]["name"] == "block-dmz-to-trust"

    def test_implicit_deny_after_removing_default(self, fw):
        fw.delete_rule("rule-006")  # 删除 default-deny
        fw.commit()
        r = fw.test_traffic("untrust", "trust", "8.8.8.8", "10.1.5.5", "tcp", "3389")
        assert r["matched"] is False
        assert r["action"] == "deny"

    def test_commit_then_traffic_hits_new_rule(self, fw):
        rid = _add_ok(fw)["rule"]["rule_id"]
        fw.commit()
        r = fw.test_traffic("trust", "dmz", "10.1.8.5", "172.16.1.20", "tcp", "22")
        assert r["matched"] is True
        assert r["matched_rule"]["rule_id"] == rid
        assert r["action"] == "allow"

    def test_test_traffic_invalid_zone(self, fw):
        r = fw.test_traffic("office", "dmz", "1.1.1.1", "2.2.2.2", "tcp", "80")
        assert r["success"] is False


# ---------- 故障注入 ----------

class TestFaultInjection:
    def test_commit_reject(self, fw):
        _add_ok(fw)
        fw.fault = {"mode": "commit_reject", "fail_times": 0}
        assert fw.commit()["success"] is False
        assert fw.commit()["success"] is False
        assert fw.running_revision == 1  # 从未生效
        assert fw.diff()["has_changes"] is True  # 候选保留

    def test_commit_flaky_recovers(self, fw):
        _add_ok(fw)
        fw.fault = {"mode": "commit_flaky", "fail_times": 2}
        assert fw.commit()["success"] is False
        assert fw.commit()["success"] is False
        r = fw.commit()
        assert r["success"] is True
        assert fw.running_revision == 2

    def test_commit_lose_applies_but_reports_failure(self, fw):
        rid = _add_ok(fw)["rule"]["rule_id"]
        fw.fault = {"mode": "commit_lose", "fail_times": 0}
        r = fw.commit()
        assert r["success"] is False           # 设备未确认
        assert "hint" in r                     # 提示 Agent 核实状态
        assert fw.running_revision == 2        # 实际已生效
        assert any(x["rule_id"] == rid for x in fw.running_rules)


# ---------- 审计日志 ----------

class TestAudit:
    def test_audit_log_records_ops(self, fw):
        _add_ok(fw)
        fw.add_rule("bad", "office", "dmz", "any", "any", "tcp", "22", "allow", "")
        fw.commit()
        ops = [(e["operation"], e["result"]) for e in fw.audit_log]
        assert ("add_rule", "success") in ops
        assert ("add_rule", "error") in ops
        assert ("commit", "success") in ops
        assert fw.audit_log[-1]["running_revision"] == 2
