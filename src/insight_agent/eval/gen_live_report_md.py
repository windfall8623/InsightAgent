"""Render the live 200-case benchmark JSON into a Markdown evaluation report.

Reads docs/cn_retail_200_live_benchmark_report.json and writes
docs/cn_retail_200_live_evaluation_report.md. Numbers are copied verbatim from
the JSON so the report can never drift from the measured run.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path("docs/cn_retail_200_live_benchmark_report.json")
OUT = Path("docs/cn_retail_200_live_evaluation_report.md")

STRAT_CN = {
    "cheap_flat": "无检索 + 快路由(最省)",
    "cheap_hybrid": "混合检索 + 快路由(基线)",
    "rich_vector": "纯向量检索 + 快路由",
    "deep_hybrid": "混合检索 + 恒定大模型",
}
TIER_CN = {
    "Tier_1": "Tier 1 单表基础(60)",
    "Tier_2": "Tier 2 跨2~3表 JOIN(70)",
    "Tier_3": "Tier 3 跨4表复杂诊断(50)",
    "Tier_4": "Tier 4 红队对抗/澄清(20)",
}


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    meta = data["benchmark_meta"]
    tiers = data["tier_performance"]
    unis = data["uniform_policies"]
    cand = data["candidate_policy"]
    gate = data["sage_gate_verdict"]
    dyn = data.get("improvement_dynamics", {})

    lines: list[str] = []
    a = lines.append

    a("# CN-Retail 2.0 真实 LLM 全链路 200 题评测报告")
    a("")
    a(f"> **评测模式**:`{meta.get('mode')}`(深链路由 **{meta.get('llm_model')}** 真实驱动)  ")
    a(f"> **数据底座**:{meta.get('database_rows'):,} 行 · {meta.get('database_tables')} 表 · {meta.get('total_cases')} 题 × {len(meta.get('strategies_tested', []))} 策略 = {meta.get('total_executions')} 次执行  ")
    a(f"> **报告生成**:{date.today().isoformat()} · 数值全部取自 `{SRC.as_posix()}`,与实测 JSON 严格一致  ")
    a("")
    a("---")
    a("")

    # Headline
    base_name = meta.get("baseline_strategy")
    a("## 一、总览(TL;DR)")
    a("")
    a(f"- 本次为**真实接通 LLM(`{meta.get('llm_model')}`)的全链路评测**:简单题由确定性编译器兜底,复杂题真实走大模型生成 SQL。")
    a(f"- 共 **{meta.get('total_executions')}** 次执行(4 策略 × {meta.get('total_cases')} 题),基线策略为 `{base_name}`。")
    cand_m = cand["candidate"]
    base_m = cand["baseline"]
    a(f"- 自进化候选策略成功率 **{pct(cand_m['success'])}**,基线 **{pct(base_m['success'])}**,复合效用改善 **{cand.get('improvement_pct', 0):+.2f}%**。")
    a(f"- SAGE 统计门禁判定:**{gate.get('decision', '').upper()}**(p={gate.get('p_value')},改进 {gate.get('improvements')} / 回退 {gate.get('regressions')} / 不变 {gate.get('unchanged')})。")
    a("")
    a("---")
    a("")

    # Strategy table
    a("## 二、四策略全矩阵实测(每策略 200 题)")
    a("")
    a("| 策略 | 形态 | 成功率 | 检索召回 | 总成本($) | 平均延迟(ms) | 复合效用分 |")
    a("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for name in meta.get("strategies_tested", []):
        m = unis.get(name, {})
        tag = " **(基线)**" if name == base_name else ""
        a(f"| `{name}`{tag} | {STRAT_CN.get(name, name)} | {pct(m.get('success', 0))} | {pct(m.get('recall', 0))} | {m.get('cost', 0):.5f} | {m.get('latency', 0):.2f} | {m.get('score', 0):.2f} |")
    a(f"| **候选策略** | 形态感知动态分流 | **{pct(cand_m['success'])}** | **{pct(cand_m['recall'])}** | **{cand_m['cost']:.5f}** | **{cand_m['latency']:.2f}** | **{cand_m['score']:.2f}** |")
    a("")
    a("---")
    a("")

    # Tier breakdown (candidate policy)
    a("## 三、候选策略分层通过率(Tier Breakdown)")
    a("")
    a("| 梯队 | 通过率 | 平均召回 | 平均延迟(ms) |")
    a("| :--- | :---: | :---: | :---: |")
    for key in ["Tier_1", "Tier_2", "Tier_3", "Tier_4"]:
        t = tiers.get(key, {})
        a(f"| {TIER_CN.get(key, key)} | {pct(t.get('pass_rate', 0))} | {pct(t.get('avg_recall', 0))} | {t.get('avg_latency_ms', 0):.2f} |")
    a("")
    a("---")
    a("")

    # SAGE
    a("## 四、SAGE 统计准入门禁")
    a("")
    a(f"- **判定**:`{gate.get('decision', '').upper()}`")
    a(f"- **逐条对比**:改进 {gate.get('improvements')} 对 · 回退 {gate.get('regressions')} 对 · 不变 {gate.get('unchanged')} 对 · 净 {gate.get('net')}")
    a(f"- **McNemar 精确检验 p 值**:{gate.get('p_value')}")
    a(f"- **门禁结论说明**:{gate.get('reason')}")
    reg = gate.get("regressions_by_id") or []
    imp = gate.get("improvements_by_id") or []
    if reg:
        a(f"- **捕获的回退用例**:{', '.join(reg)}")
    if imp:
        a(f"- **确认的改进用例**:{', '.join(imp)}")
    a(f"- **候选是否被接受**:{'是' if cand.get('accepted') else '否'}")
    if cand.get("reasons"):
        a(f"  - 原因:{'; '.join(cand['reasons'])}")
    a("")

    # Shape distribution
    sd = cand.get("shape_distribution", {})
    if sd:
        a("### 候选策略的形态分流分布")
        a("")
        a("| 被选中的策略 | 覆盖的问题形态数 |")
        a("| :--- | :---: |")
        for k, v in sorted(sd.items(), key=lambda kv: -kv[1]):
            a(f"| `{k}` | {v} |")
        a("")
    a("---")
    a("")

    # Dynamics
    if dyn:
        a("## 五、改进动态(Improvement Dynamics)")
        a("")
        a(f"- 观测步数:{dyn.get('steps')} · 改进 {dyn.get('improvements')} · 回退 {dyn.get('regressions')}")
        a(f"- 改进率:{dyn.get('improvement_rate')} · 平均改进幅度:{dyn.get('mean_improvement')} · 平均回退幅度:{dyn.get('mean_regression')}")
        a("")

    a("---")
    a("")
    a("## 六、口径声明")
    a("")
    a(f"本报告为 `--live` 模式产物,深链路由 `{meta.get('llm_model')}` 真实生成 SQL(非纯规则)。")
    a("成功率以**结果集精确匹配**(gold_sql 与模型 SQL 在 SQLite 上执行后行集一致)判定,")
    a("非 LLM 主观打分;red-team 题以是否正确拦截/澄清判定。所有数值可由上述 JSON 复现。")
    a("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Wrote {OUT.resolve()} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
