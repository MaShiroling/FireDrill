"""防火墙状态持久化仓储。

SQLite 只负责状态与事务语义，业务校验仍由 ``FirewallState`` 完成。表结构显式保留
Running Revision、Candidate ChangeSet、审计日志与 Commit 幂等结果，便于重启恢复和评测追溯。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import Integer, String, Text, UniqueConstraint, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class RevisionConflict(RuntimeError):
    """候选配置基于的 Running Revision 已过期。"""


class ChangeSetConflict(RuntimeError):
    """候选 ChangeSet 已被其他提交或回滚替换。"""


class FirewallRepository(Protocol):
    """``FirewallState`` 所依赖的最小持久化接口。"""

    def load(self) -> dict[str, Any] | None: ...

    def reset(self, state: dict[str, Any]) -> dict[str, Any]: ...

    def save_working(self, state: dict[str, Any], audit_event: dict[str, Any]) -> None: ...

    def get_commit_result(self, idempotency_key: str) -> dict[str, Any] | None: ...

    def commit(
        self,
        state: dict[str, Any],
        expected_revision: int,
        idempotency_key: str,
        response: dict[str, Any],
        audit_event: dict[str, Any],
    ) -> str: ...

    def discard(self, state: dict[str, Any], audit_event: dict[str, Any]) -> str: ...


class Base(DeclarativeBase):
    pass


class DeviceRow(Base):
    __tablename__ = "firewall_devices"

    device_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    firmware: Mapped[str] = mapped_column(String(64), nullable=False)
    zones_json: Mapped[str] = mapped_column(Text, nullable=False)
    running_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    rule_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    fault_json: Mapped[str] = mapped_column(Text, nullable=False)
    hit_counts_json: Mapped[str] = mapped_column(Text, nullable=False)


class RevisionRow(Base):
    __tablename__ = "config_revisions"
    __table_args__ = (UniqueConstraint("device_id", "revision", name="uq_device_revision"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    rules_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_change_set_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)


class ChangeSetRow(Base):
    __tablename__ = "change_sets"

    change_set_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    base_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    candidate_rules_json: Mapped[str] = mapped_column(Text, nullable=False)
    commit_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False)


class AuditLogRow(Base):
    __tablename__ = "firewall_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    result: Mapped[str] = mapped_column(String(16), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    running_revision: Mapped[int] = mapped_column(Integer, nullable=False)


class CommitRequestRow(Base):
    __tablename__ = "commit_requests"
    __table_args__ = (
        UniqueConstraint("device_id", "idempotency_key", name="uq_device_commit_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    change_set_id: Mapped[str] = mapped_column(String(36), nullable=False)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load(value: str) -> Any:
    return json.loads(value)


class SQLiteFirewallRepository:
    """单设备 SQLite 仓储；每个公开写方法都在一个数据库事务内完成。"""

    def __init__(self, db_path: str | Path, device_id: str = "fake-fw-01") -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(self.engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        Base.metadata.create_all(self.engine)

    @staticmethod
    def _new_change_set_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _append_audit(session: Session, device_id: str, item: dict[str, Any]) -> None:
        session.add(
            AuditLogRow(
                device_id=device_id,
                timestamp=item["timestamp"],
                operation=item["operation"],
                params_json=_dump(item["params"]),
                result=item["result"],
                detail=item["detail"],
                running_revision=item["running_revision"],
            )
        )

    @staticmethod
    def _active_change_set(session: Session, device_id: str) -> ChangeSetRow | None:
        return session.scalar(
            select(ChangeSetRow)
            .where(
                ChangeSetRow.device_id == device_id,
                ChangeSetRow.status == "active",
            )
            .order_by(ChangeSetRow.created_at.desc())
        )

    def load(self) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            device = session.get(DeviceRow, self.device_id)
            if device is None:
                return None
            revision = session.scalar(
                select(RevisionRow).where(
                    RevisionRow.device_id == self.device_id,
                    RevisionRow.revision == device.running_revision,
                )
            )
            change_set = self._active_change_set(session, self.device_id)
            if revision is None or change_set is None:
                raise RuntimeError("SQLite 防火墙状态不完整：缺少 Revision 或 active ChangeSet")
            audits = session.scalars(
                select(AuditLogRow)
                .where(
                    AuditLogRow.device_id == self.device_id,
                )
                .order_by(AuditLogRow.id)
            ).all()
            return {
                "hostname": device.hostname,
                "model": device.model,
                "firmware": device.firmware,
                "zones": _load(device.zones_json),
                "running_revision": device.running_revision,
                "running_rules": _load(revision.rules_json),
                "candidate_rules": _load(change_set.candidate_rules_json),
                "candidate_base_revision": change_set.base_revision,
                "change_set_id": change_set.change_set_id,
                "rule_seq": device.rule_seq,
                "hit_counts": _load(device.hit_counts_json),
                "fault": _load(device.fault_json),
                "audit_log": [
                    {
                        "timestamp": row.timestamp,
                        "operation": row.operation,
                        "params": _load(row.params_json),
                        "result": row.result,
                        "detail": row.detail,
                        "running_revision": row.running_revision,
                    }
                    for row in audits
                ],
            }

    def reset(self, state: dict[str, Any]) -> dict[str, Any]:
        change_set_id = self._new_change_set_id()
        now = state["now"]
        with Session(self.engine) as session, session.begin():
            session.query(CommitRequestRow).filter_by(device_id=self.device_id).delete()
            session.query(AuditLogRow).filter_by(device_id=self.device_id).delete()
            session.query(ChangeSetRow).filter_by(device_id=self.device_id).delete()
            session.query(RevisionRow).filter_by(device_id=self.device_id).delete()
            session.query(DeviceRow).filter_by(device_id=self.device_id).delete()
            session.add(
                DeviceRow(
                    device_id=self.device_id,
                    hostname=state["hostname"],
                    model=state["model"],
                    firmware=state["firmware"],
                    zones_json=_dump(state["zones"]),
                    running_revision=state["running_revision"],
                    rule_seq=state["rule_seq"],
                    fault_json=_dump(state["fault"]),
                    hit_counts_json=_dump(state["hit_counts"]),
                )
            )
            session.add(
                RevisionRow(
                    device_id=self.device_id,
                    revision=state["running_revision"],
                    rules_json=_dump(state["running_rules"]),
                    source_change_set_id=None,
                    created_at=now,
                )
            )
            session.add(
                ChangeSetRow(
                    change_set_id=change_set_id,
                    device_id=self.device_id,
                    base_revision=state["running_revision"],
                    status="active",
                    candidate_rules_json=_dump(state["candidate_rules"]),
                    commit_key=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        return {
            "change_set_id": change_set_id,
            "candidate_base_revision": state["running_revision"],
        }

    def _update_device(self, device: DeviceRow, state: dict[str, Any]) -> None:
        device.hostname = state["hostname"]
        device.model = state["model"]
        device.firmware = state["firmware"]
        device.zones_json = _dump(state["zones"])
        device.rule_seq = state["rule_seq"]
        device.fault_json = _dump(state["fault"])
        device.hit_counts_json = _dump(state["hit_counts"])

    def save_working(self, state: dict[str, Any], audit_event: dict[str, Any]) -> None:
        with Session(self.engine) as session, session.begin():
            device = session.get(DeviceRow, self.device_id)
            change_set = session.get(ChangeSetRow, state["change_set_id"])
            if device is None or change_set is None or change_set.status != "active":
                raise ChangeSetConflict("候选 ChangeSet 已失效，请重新读取设备状态")
            self._update_device(device, state)
            change_set.candidate_rules_json = _dump(state["candidate_rules"])
            change_set.updated_at = audit_event["timestamp"]
            self._append_audit(session, self.device_id, audit_event)

    def get_commit_result(self, idempotency_key: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.scalar(
                select(CommitRequestRow).where(
                    CommitRequestRow.device_id == self.device_id,
                    CommitRequestRow.idempotency_key == idempotency_key,
                )
            )
            return _load(row.response_json) if row else None

    def commit(
        self,
        state: dict[str, Any],
        expected_revision: int,
        idempotency_key: str,
        response: dict[str, Any],
        audit_event: dict[str, Any],
    ) -> str:
        new_change_set_id = self._new_change_set_id()
        with Session(self.engine) as session, session.begin():
            previous = session.scalar(
                select(CommitRequestRow).where(
                    CommitRequestRow.device_id == self.device_id,
                    CommitRequestRow.idempotency_key == idempotency_key,
                )
            )
            if previous is not None:
                return previous.change_set_id

            device = session.get(DeviceRow, self.device_id)
            if device is None:
                raise RuntimeError("防火墙设备状态不存在")
            if device.running_revision != expected_revision:
                raise RevisionConflict(
                    f"版本冲突: Candidate 基于 R{expected_revision}，"
                    f"当前 Running 已是 R{device.running_revision}"
                )
            change_set = session.get(ChangeSetRow, state["change_set_id"])
            if change_set is None or change_set.status != "active":
                raise ChangeSetConflict("候选 ChangeSet 已被提交或回滚")

            new_revision = expected_revision + 1
            self._update_device(device, state)
            device.running_revision = new_revision
            change_set.status = "committed"
            change_set.commit_key = idempotency_key
            change_set.updated_at = audit_event["timestamp"]
            session.add(
                RevisionRow(
                    device_id=self.device_id,
                    revision=new_revision,
                    rules_json=_dump(state["candidate_rules"]),
                    source_change_set_id=change_set.change_set_id,
                    created_at=audit_event["timestamp"],
                )
            )
            session.add(
                ChangeSetRow(
                    change_set_id=new_change_set_id,
                    device_id=self.device_id,
                    base_revision=new_revision,
                    status="active",
                    candidate_rules_json=_dump(state["candidate_rules"]),
                    commit_key=None,
                    created_at=audit_event["timestamp"],
                    updated_at=audit_event["timestamp"],
                )
            )
            session.add(
                CommitRequestRow(
                    device_id=self.device_id,
                    idempotency_key=idempotency_key,
                    change_set_id=new_change_set_id,
                    response_json=_dump(response),
                    created_at=audit_event["timestamp"],
                )
            )
            self._append_audit(session, self.device_id, audit_event)
        return new_change_set_id

    def discard(self, state: dict[str, Any], audit_event: dict[str, Any]) -> str:
        new_change_set_id = self._new_change_set_id()
        with Session(self.engine) as session, session.begin():
            device = session.get(DeviceRow, self.device_id)
            change_set = session.get(ChangeSetRow, state["change_set_id"])
            if device is None or change_set is None or change_set.status != "active":
                raise ChangeSetConflict("候选 ChangeSet 已被提交或回滚")
            if device.running_revision != state["candidate_base_revision"]:
                raise RevisionConflict(
                    f"版本冲突: Candidate 基于 R{state['candidate_base_revision']}，"
                    f"当前 Running 已是 R{device.running_revision}"
                )
            self._update_device(device, state)
            change_set.status = "discarded"
            change_set.updated_at = audit_event["timestamp"]
            session.add(
                ChangeSetRow(
                    change_set_id=new_change_set_id,
                    device_id=self.device_id,
                    base_revision=device.running_revision,
                    status="active",
                    candidate_rules_json=_dump(state["running_rules"]),
                    commit_key=None,
                    created_at=audit_event["timestamp"],
                    updated_at=audit_event["timestamp"],
                )
            )
            self._append_audit(session, self.device_id, audit_event)
        return new_change_set_id
