"""SQLite 防火墙仓储的重启恢复、事务、幂等与乐观锁测试。"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_servers"))

from firewall_repository import SQLiteFirewallRepository
from firewall_server import FirewallState


def _state(db_path: Path) -> FirewallState:
    return FirewallState(SQLiteFirewallRepository(db_path))


def _add_ok(fw: FirewallState, name: str = "persisted-ssh") -> dict:
    return fw.add_rule(
        name, "trust", "dmz", "10.1.9.0/24", "172.16.1.30/32", "tcp", "22", "allow", "持久化测试"
    )


def test_candidate_and_audit_survive_restart(tmp_path):
    db_path = tmp_path / "firewall.db"
    first = _state(db_path)
    added = _add_ok(first)

    restarted = _state(db_path)

    assert restarted.running_revision == 1
    assert restarted.diff()["has_changes"] is True
    assert any(rule["rule_id"] == added["rule"]["rule_id"] for rule in restarted.candidate_rules)
    assert restarted.audit_log[0]["operation"] == "add_rule"


def test_commit_creates_revision_and_new_active_changeset(tmp_path):
    db_path = tmp_path / "firewall.db"
    fw = _state(db_path)
    old_change_set = fw.change_set_id
    rule_id = _add_ok(fw)["rule"]["rule_id"]

    result = fw.commit(idempotency_key="commit-001", expected_revision=1)
    restarted = _state(db_path)

    assert result["success"] is True
    assert restarted.running_revision == 2
    assert restarted.candidate_base_revision == 2
    assert restarted.change_set_id != old_change_set
    assert restarted.candidate_rules == restarted.running_rules
    assert any(rule["rule_id"] == rule_id for rule in restarted.running_rules)

    with sqlite3.connect(db_path) as connection:
        revision_count = connection.execute("SELECT COUNT(*) FROM config_revisions").fetchone()[0]
        statuses = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM change_sets GROUP BY status"
            ).fetchall()
        )
    assert revision_count == 2
    assert statuses == {"active": 1, "committed": 1}


def test_commit_idempotency_returns_original_result(tmp_path):
    fw = _state(tmp_path / "firewall.db")
    _add_ok(fw)

    first = fw.commit(idempotency_key="same-request")
    replay = fw.commit(idempotency_key="same-request")

    assert replay == first
    assert fw.running_revision == 2
    with sqlite3.connect(tmp_path / "firewall.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM config_revisions").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM commit_requests").fetchone()[0] == 1


def test_stale_candidate_commit_is_rejected(tmp_path):
    db_path = tmp_path / "firewall.db"
    stale = _state(db_path)
    _add_ok(stale)
    winner = _state(db_path)
    assert winner.commit(idempotency_key="winner")["success"] is True

    result = stale.commit(idempotency_key="stale")

    assert result["success"] is False
    assert result["conflict"] is True
    assert "版本冲突" in result["error"]
    assert stale.running_revision == 2


def test_discard_survives_restart_and_keeps_revision(tmp_path):
    db_path = tmp_path / "firewall.db"
    fw = _state(db_path)
    old_change_set = fw.change_set_id
    _add_ok(fw)

    assert fw.discard()["success"] is True
    restarted = _state(db_path)

    assert restarted.running_revision == 1
    assert restarted.candidate_rules == restarted.running_rules
    assert restarted.change_set_id != old_change_set
    with sqlite3.connect(db_path) as connection:
        statuses = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM change_sets GROUP BY status"
            ).fetchall()
        )
    assert statuses == {"active": 1, "discarded": 1}
