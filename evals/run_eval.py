"""防火墙 Agent 评测 runner

用法：
    NO_PROXY=localhost,127.0.0.1 .venv/bin/python evals/run_eval.py --runs 3 --tag current
    NO_PROXY=localhost,127.0.0.1 .venv/bin/python evals/run_eval.py --runs 3 --tag legacy --legacy --categories change,delete,modify

每条用例流程：reset 出厂 → 注入故障(可选) → 跑 plan-execute-replan → 拉 snapshot → 断言评分。
结果写 evals/results/<tag>.jsonl，末尾打印汇总。
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.evaluation import evaluate_run, load_trace  # noqa: E402

FW_BASE = "http://127.0.0.1:8005"
CASES_FILE = ROOT / "evals" / "cases_firewall.json"
RESULTS_DIR = ROOT / "evals" / "results"


async def run_case(case: dict, run_idx: int, timeout: int, tag: str, legacy: bool) -> dict:
    from fastmcp import Client

    from app.services.aiops_service import aiops_service

    async with httpx.AsyncClient(base_url=FW_BASE, timeout=15) as h:
        await h.post("/admin/reset", json={})
        if case.get("scenario"):
            await h.post("/admin/scenario", json=case["scenario"])

    report, error_msg, steps = "", "", 0
    trace_id, trace_path = None, None
    t0 = time.time()
    session = f"eval-{case['id']}-r{run_idx}-{int(t0)}"
    try:

        async def _drive():
            nonlocal report, steps, trace_id, trace_path
            async for ev in aiops_service.execute(
                case["task"],
                session_id=session,
                trace_metadata={
                    "source": "firewall_eval",
                    "case_id": case["id"],
                    "category": case["category"],
                    "run": run_idx,
                    "tag": tag,
                    "legacy": legacy,
                    "scenario": case.get("scenario"),
                    "flywheel": case.get("flywheel"),
                },
            ):
                if ev.get("type") == "trace_started":
                    trace_id = ev.get("trace_id")
                    trace_path = ev.get("trace_path")
                elif ev.get("type") == "step_complete":
                    steps += 1
                elif ev.get("type") == "report":
                    report = ev.get("report", "")
                elif ev.get("type") == "error":
                    trace_id = ev.get("trace_id")
                    trace_path = ev.get("trace_path")
                    raise RuntimeError(ev.get("message", "unknown"))
                elif ev.get("type") == "complete":
                    trace_id = ev.get("trace_id")
                    trace_path = ev.get("trace_path")

        await asyncio.wait_for(_drive(), timeout=timeout)
    except Exception as e:
        error_msg = str(e)[:300]

    async with httpx.AsyncClient(base_url=FW_BASE, timeout=15) as h:
        snapshot = (await h.get("/admin/snapshot")).json()

    trace = load_trace(trace_path, root=ROOT)

    async def _evaluate(traffic_probe=None):
        return await evaluate_run(
            assertions=case["assert"],
            snapshot=snapshot,
            report=report,
            expect_success=case["expect_success"],
            run_error=error_msg,
            steps=steps,
            trace=trace,
            traffic_probe=traffic_probe,
        )

    if any(assertion["type"] == "traffic" for assertion in case["assert"]):
        try:
            async with Client(f"{FW_BASE}/mcp") as fw_client:

                async def traffic_probe(packet: dict) -> dict:
                    response = await fw_client.call_tool("test_traffic", packet)
                    return json.loads(response.content[0].text)

                evaluation = await _evaluate(traffic_probe)
        except Exception as e:
            # 评估通道异常只会使 traffic 断言失败，不覆盖其他终态断言。
            evaluation = await _evaluate()
            evaluation.failure_evidence["evaluator_error"].append(str(e)[:200])
    else:
        evaluation = await _evaluate()

    results = evaluation.assertion_results
    evaluation_data = evaluation.to_dict()
    return {
        "case_id": case["id"],
        "category": case["category"],
        "run": run_idx,
        "expect_success": case["expect_success"],
        "passed": evaluation.success,
        "claims_success": evaluation.claims_success,
        "claims_failure": evaluation.claims_failure,
        "fake_complete": evaluation.fake_completion,
        "false_failure": evaluation.false_failure,
        "correct_failure": evaluation.correct_failure,
        "steps": steps,
        "duration_s": round(time.time() - t0, 1),
        "trace_id": trace_id,
        "trace_path": trace_path,
        "error": error_msg,
        "asserts": [
            {
                "type": result.assertion["type"],
                "pass": result.passed,
                "detail": result.detail,
            }
            for result in results
        ],
        "failure_codes": evaluation_data["failure_codes"],
        "failure_evidence": evaluation_data["failure_evidence"],
        "evaluation": evaluation_data,
        "flywheel": case.get("flywheel"),
        "report_tail": report[-400:],
    }


def aggregate(records: list) -> None:
    total = len(records)
    passed = sum(r["passed"] for r in records)
    fake = sum(r["fake_complete"] for r in records)
    print("\n" + "=" * 60)
    print(f"总运行: {total}  |  任务成功率: {passed}/{total} = {passed/total:.1%}")
    print(f"假完成（声称成功但断言失败）: {fake}/{total} = {fake/total:.1%}")
    # 错误类用例的"正确失败"率
    neg = [r for r in records if not r["expect_success"]]
    if neg:
        cf = sum(r["correct_failure"] for r in neg)
        print(f"错误类用例正确失败率（不声称成功）: {cf}/{len(neg)} = {cf/len(neg):.1%}")
    print("-" * 60)
    print("按类别:")
    cats = sorted({r["category"] for r in records})
    for c in cats:
        sub = [r for r in records if r["category"] == c]
        p = sum(r["passed"] for r in sub)
        f = sum(r["fake_complete"] for r in sub)
        print(f"  {c:10s} 成功率 {p}/{len(sub)} = {p/len(sub):.0%}   假完成 {f}/{len(sub)}")
    print("-" * 60)
    print("逐用例（多次运行的通过次数）:")
    ids = sorted({r["case_id"] for r in records})
    for cid in ids:
        sub = [r for r in records if r["case_id"] == cid]
        p = sum(r["passed"] for r in sub)
        f = sum(r["fake_complete"] for r in sub)
        print(f"  {cid}: 通过 {p}/{len(sub)}" + (f"  假完成 {f}" if f else ""))

    failure_counts: dict[str, int] = {}
    for record in records:
        for code in record.get("failure_codes", []):
            failure_counts[code] = failure_counts.get(code, 0) + 1
    if failure_counts:
        print("-" * 60)
        print("失败分类（同一次运行可命中多个标签）:")
        for code, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {code:26s} {count}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case-file",
        default=str(CASES_FILE),
        help="用例 JSON 数组；可传入数据飞轮生成的 replay_cases.json",
    )
    ap.add_argument("--cases", default="", help="逗号分隔用例 ID，默认全部")
    ap.add_argument("--categories", default="", help="逗号分隔类别过滤")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument("--legacy", action="store_true", help="启用 REPLANNER_LEGACY 对照模式")
    args = ap.parse_args()

    if args.legacy:
        os.environ["REPLANNER_LEGACY"] = "1"
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    cases = json.loads(Path(args.case_file).read_text(encoding="utf-8"))
    if args.cases:
        keep = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in keep]
    if args.categories:
        keep = set(args.categories.split(","))
        cases = [c for c in cases if c["category"] in keep]
    print(f"用例数: {len(cases)} × {args.runs} 轮, tag={args.tag}, legacy={args.legacy}")

    async with httpx.AsyncClient(base_url=FW_BASE, timeout=5) as h:
        r = await h.get("/admin/health")
        assert r.json()["status"] == "ok", "防火墙服务未就绪"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{args.tag}.jsonl"

    # 断点续跑：已存在的 (case_id, run) 直接跳过
    done = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["case_id"], r["run"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        if done:
            print(f"续跑：跳过已完成的 {len(done)} 条")

    records = []
    for run_idx in range(1, args.runs + 1):
        for case in cases:
            if (case["id"], run_idx) in done:
                continue
            print(f"\n[{args.tag}] run{run_idx} {case['id']} ({case['category']}) ...", flush=True)
            rec = await run_case(case, run_idx, args.timeout, args.tag, args.legacy)
            records.append(rec)
            with out_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mark = "PASS" if rec["passed"] else ("FAKE" if rec["fake_complete"] else "FAIL")
            print(
                f"  -> {mark}  steps={rec['steps']} {rec['duration_s']}s "
                + ("" if rec["passed"] else str([a for a in rec["asserts"] if not a["pass"]])[:200])
            )
            await asyncio.sleep(2)  # 缓一下，防限流

    # 汇总时纳入文件中的全部历史记录（含续跑前完成的）
    all_records = []
    for line in out_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    aggregate(all_records)


if __name__ == "__main__":
    asyncio.run(main())
