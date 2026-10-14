"""Reproducible offline intent ablations with separate development and audit sets.

Run: PYTHONPATH=src python -m insight_agent.eval.intent_ablation \
    --output docs/intent_ablation_report.json \
    --markdown docs/intent_ablation_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.app.router import IntentFeatures, classify_intent
from insight_agent.eval.intent_mode_compare import INTENTS, PROBES


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    mode: str = "cascade"
    features: IntentFeatures = field(default_factory=IntentFeatures)
    compare_to: str | None = "cascade_full"


VARIANTS = (
    Variant("rule_legacy", "旧关键词基线；只能输出诊断、闲聊、简单指标", mode="rule", compare_to=None),
    Variant("jev_local", "本地关键词计分 + softmax；不是远程 Jev 或 LLM", mode="jev", compare_to=None),
    Variant("cascade_full", "当前完整级联", compare_to=None),
    Variant("no_injection_precheck", "仅移除级联最前面的注入预检", features=IntentFeatures(injection_precheck=False)),
    Variant("no_diagnostic_fastpath", "仅移除明确诊断快车道", features=IntentFeatures(diagnostic_fast_path=False)),
    Variant("no_greeting_fastpath", "仅移除问候快车道", features=IntentFeatures(greeting_fast_path=False)),
    Variant("no_nonbusiness_shortcut", "仅移除无业务词直接闲聊的短路", features=IntentFeatures(non_business_shortcut=False)),
    Variant("no_local_fallback", "移除末端判别器；未命中规则时默认简单指标", features=IntentFeatures(fallback=False)),
    Variant("no_aggregation_guard", "仅移除本地判别器的聚合护栏", features=IntentFeatures(aggregation_guard=False)),
    Variant("jev_no_shared_patterns", "Jev 保留原有注入原型，不使用共享扩展词表", mode="jev", features=IntentFeatures(shared_injection_patterns=False), compare_to="jev_local"),
    Variant("cascade_no_shared_patterns", "级联保留前置预检，仅移除本地共享词表", features=IntentFeatures(shared_injection_patterns=False)),
    Variant("no_precheck_or_shared_patterns", "交互项：同时移除前置预检与本地共享词表，保留原注入原型", features=IntentFeatures(injection_precheck=False, shared_injection_patterns=False)),
)

DATA_NOTES = {
    "development_100": "100条合成开发/回归题；曾用于规则调优，不是独立测试集；coref标签仅传入末轮文本。",
    "benign_controls": "冻结分类逻辑后独立编写的合法请求负对照；用于检查过度拦截，不代表生产分布。",
    "context_last_turn": "新编完整对话，但模拟当前入口，仅将末轮文本交给classify_intent；gold根据完整历史标注。",
    "context_resolved": "探索性接线：先使用现有CoreferenceResolver处理历史，再分类；没有修改API接线。",
}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def gate_metrics(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    def positive(row: dict[str, Any]) -> bool:
        return bool(row["is_anomaly"]) if label == "diagnostic_anomaly" else row["pred"] == label

    tp = sum(row["gold"] == label and positive(row) for row in rows)
    fp = sum(row["gold"] != label and positive(row) for row in rows)
    fn = sum(row["gold"] == label and not positive(row) for row in rows)
    tn = len(rows) - tp - fp - fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": tp + fn,
        "precision": precision, "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": _ratio(fp, fp + tn),
        "precision_defined": bool(tp + fp),
        "recall_defined": bool(tp + fn),
    }


def intent_route(intent: str) -> str:
    """Four-way label score, preserving injection separately from analytics."""
    return "analytics" if intent in ("simple_metric", "complex_analytic") else intent


def api_dispatch(intent: str, is_anomaly: bool) -> str:
    """Mirror /ask's dispatch order, not its downstream safety outcome."""
    if intent == "chat_general":
        return "chat"
    return "diagnostic" if is_anomaly else "orchestrator"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    raw_correct = sum(row["gold"] == row["pred"] for row in rows)
    route_correct = sum(intent_route(row["gold"]) == intent_route(row["pred"]) for row in rows)
    dispatch_correct = sum(
        api_dispatch(row["gold"], row["gold"] == "diagnostic_anomaly")
        == api_dispatch(row["pred"], row["is_anomaly"])
        for row in rows
    )
    per_class = {label: gate_metrics(rows, label) for label in INTENTS}
    tags = sorted({row["tag"] for row in rows})
    confusion = {
        gold: {pred: sum(row["gold"] == gold and row["pred"] == pred for row in rows) for pred in INTENTS}
        for gold in INTENTS
    }
    supported = [value["f1"] for value in per_class.values() if value["support"]]
    return {
        "n": n,
        "raw_correct": raw_correct,
        "raw_accuracy": _ratio(raw_correct, n),
        "intent_route_correct": route_correct,
        "intent_route_accuracy": _ratio(route_correct, n),
        "api_dispatch_correct": dispatch_correct,
        "api_dispatch_agreement": _ratio(dispatch_correct, n),
        "macro_f1_supported": _ratio(sum(supported), len(supported)),
        "per_class": per_class,
        "per_tag": {
            tag: {
                "n": sum(row["tag"] == tag for row in rows),
                "correct": sum(row["tag"] == tag and row["gold"] == row["pred"] for row in rows),
                "errors": [row["id"] for row in rows if row["tag"] == tag and row["gold"] != row["pred"]],
            } for tag in tags
        },
        "confusion": confusion,
        "decision_modes": dict(Counter(row["decision_mode"] for row in rows)),
        "errors": [row["id"] for row in rows if row["gold"] != row["pred"]],
        "rows": rows,
    }


def compare_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = {row["id"]: row for row in baseline["rows"]}
    current = {row["id"]: row for row in candidate["rows"]}
    if base.keys() != current.keys():
        raise ValueError("Paired comparisons require identical case IDs")
    gains = [id_ for id_, row in current.items() if row["pred"] == row["gold"] and base[id_]["pred"] != base[id_]["gold"]]
    losses = [id_ for id_, row in current.items() if row["pred"] != row["gold"] and base[id_]["pred"] == base[id_]["gold"]]
    changed = [id_ for id_, row in current.items() if row["pred"] != base[id_]["pred"]]
    return {
        "raw_delta_pp": 100 * (candidate["raw_accuracy"] - baseline["raw_accuracy"]),
        "intent_route_delta_pp": 100 * (candidate["intent_route_accuracy"] - baseline["intent_route_accuracy"]),
        "improved": gains,
        "regressed": losses,
        "changed_predictions": changed,
    }


def evaluate_variant(cases: Sequence[dict[str, Any]], variant: Variant, *, resolve_history: bool = False) -> dict[str, Any]:
    rows = []
    for case in cases:
        question = case["question"]
        resolved = None
        if resolve_history:
            from insight_agent.app.coreference import CoreferenceResolver

            resolved = CoreferenceResolver().resolve(question, history=case.get("history"))
            question = resolved.resolved_query
        decision = classify_intent(question, mode=variant.mode, features=variant.features, local_only=True)
        rows.append({
            "id": case["id"], "question": case["question"], "input_question": question,
            "history": case.get("history", []), "gold": case["gold"], "tag": case["tag"],
            "rationale": case.get("rationale", ""), "pred": decision.intent,
            "is_anomaly": decision.is_anomaly, "confidence": decision.confidence,
            "probabilities": decision.probabilities, "decision_mode": decision.mode, "reason": decision.reason,
            "resolution": asdict(resolved) if resolved else None,
        })
    return summarize(rows)


def _validate_cases(cases: Sequence[dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("An evaluation set must not be empty")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if case["gold"] not in INTENTS or not case["question"].strip():
            raise ValueError(f"Invalid case: {case['id']}")
        for message in case.get("history", []):
            if message["role"] not in ("user", "assistant") or not message["content"].strip():
                raise ValueError(f"Invalid history: {case['id']}")


def _digest(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def run_experiment(validation_path: Path | None = None) -> dict[str, Any]:
    development = [
        {"id": f"dev-{i:03d}", "question": question, "gold": gold, "tag": tag, "history": []}
        for i, (question, gold, tag) in enumerate(PROBES, 1)
    ]
    validation_path = validation_path or Path(__file__).with_name("intent_validation_cases.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    datasets = {
        "development_100": development,
        "benign_controls": validation["benign_controls"],
        "context_last_turn": validation["context_dialogues"],
        "context_resolved": validation["context_dialogues"],
    }
    development_questions = {case["question"] for case in development}
    audit_questions = {case["question"] for case in validation["benign_controls"] + validation["context_dialogues"]}
    if development_questions & audit_questions:
        raise ValueError("Audit questions must not duplicate development questions")
    results = {}
    for name, cases in datasets.items():
        _validate_cases(cases)
        if name == "benign_controls" and any(case["gold"] == "prompt_injection" for case in cases):
            raise ValueError("Negative controls must not contain attacks")
        scores = {variant.name: evaluate_variant(cases, variant, resolve_history=name == "context_resolved") for variant in VARIANTS}
        repeat = {variant.name: evaluate_variant(cases, variant, resolve_history=name == "context_resolved") for variant in VARIANTS}
        if scores != repeat:
            raise AssertionError(f"Non-deterministic predictions in {name}")
        comparisons = {
            variant.name: {"baseline": variant.compare_to, **compare_rows(scores[variant.compare_to], scores[variant.name])}
            for variant in VARIANTS if variant.compare_to is not None
        }
        results[name] = {
            "note": DATA_NOTES[name], "n": len(cases), "dataset_sha256": _digest(cases),
            "label_counts": dict(Counter(case["gold"] for case in cases)),
            "variants": scores, "ablations": comparisons,
        }
    root = Path(__file__).resolve().parents[3]
    sources = (
        "src/insight_agent/app/router.py", "src/insight_agent/app/jev_classifier.py",
        "src/insight_agent/app/coreference.py", "src/insight_agent/eval/intent_mode_compare.py",
        "src/insight_agent/eval/intent_ablation.py", "src/insight_agent/eval/intent_validation_cases.json",
    )
    return {
        "schema_version": 1,
        "backend": "offline_local_keyword_scoring",
        "python": platform.python_version(),
        "reproducibility": {"repeats": 2, "exact_match": True, "latency_compared": False},
        "metric_notes": {
            "intent_route_accuracy": "四类标签一致性：simple/complex合并，但保留injection；不是端到端执行成功率。",
            "api_dispatch_agreement": "仅模拟/ask的chat/diagnostic/orchestrator分支；注入也进入orchestrator，不能证明安全拦截。",
            "confidence": "规则常量或关键词softmax分值，未经校准，不是正确率。",
            "zero_support": "无正样本时召回率未定义；以recall_defined标记，不当作0%召回。",
        },
        "limitations": [
            "开发集参与过调参，不作独立泛化或显著性推断。",
            "验证样本为独立编写的合成小样本，不是生产流量或人工盲标。",
            "rule旧基线只能输出3类；当前Jev本地和cascade共享注入规则，不是独立LLM消融。",
            "上下文消解只是离线接线实验；/ask和/ask/stream未改为先消解再分类。",
            "未调用远程Jev或GLM，未执行注入文本中的SQL，未重跑200题SQL生成评测。",
        ],
        "source_sha256": {source: hashlib.sha256((root / source).read_bytes()).hexdigest() for source in sources},
        "variants": [{"name": v.name, "description": v.description, "mode": v.mode, "features": asdict(v.features), "compare_to": v.compare_to} for v in VARIANTS],
        "datasets": results,
    }


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pr(metrics: dict[str, Any]) -> str:
    precision = _percent(metrics["precision"]) if metrics["precision_defined"] else "N/A"
    recall = _percent(metrics["recall"]) if metrics["recall_defined"] else "N/A"
    return f"{precision} / {recall}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 意图识别：100题开发集与组件消融", "",
        "## Material Passport", "",
        "- Origin Skill: experiment-agent", "- Origin Mode: validate",
        "- Verification Status: VERIFIED（仅确定性复跑一致性）",
        "- Version Label: intent_ablation_v1", "",
        "## 实验口径", "",
        "同一冻结逻辑、相同题目、逐项关闭组件；全部使用本地判别器，强制禁止远程Jev调用。",
        "每种条件运行两遍，逐题决策、分数和汇总完全一致；不把运行次数当作更多独立样本。",
        "五类准确率与四类路由标签一致率分开列示。四类仅合并简单/复杂分析，并保留注入；它不是API安全拦截率。",
        "所有P/R表示精确率/召回率；N/A表示分母为零。", "",
    ]
    development = report["datasets"]["development_100"]["variants"]
    full = development["cascade_full"]
    local = development["jev_local"]
    local_unshared = development["jev_no_shared_patterns"]
    control = report["datasets"]["benign_controls"]["variants"]["cascade_full"]
    last_turn = report["datasets"]["context_last_turn"]["variants"]["cascade_full"]
    resolved = report["datasets"]["context_resolved"]["variants"]["cascade_full"]
    lines += [
        "## 主要发现", "",
        f"- 100题开发集：旧rule {_percent(development['rule_legacy']['raw_accuracy'])}，Jev本地 {_percent(local['raw_accuracy'])}，完整cascade {_percent(full['raw_accuracy'])}。这些不是独立测试集成绩。",
        f"- Jev本地移除共享注入词表后，准确率从{_percent(local['raw_accuracy'])}降至{_percent(local_unshared['raw_accuracy'])}；注入召回从{_percent(local['per_class']['prompt_injection']['recall'])}降至{_percent(local_unshared['per_class']['prompt_injection']['recall'])}。收益来自扩展规则，不是接通了远程语义模型。",
        f"- 完整级联移除前置注入预检后，开发集准确率为{_percent(development['no_injection_precheck']['raw_accuracy'])}；移除本地判别器后为{_percent(development['no_local_fallback']['raw_accuracy'])}；移除聚合护栏后为{_percent(development['no_aggregation_guard']['raw_accuracy'])}。",
        f"- 新增{control['n']}条合法请求负对照上，完整级联准确率{_percent(control['raw_accuracy'])}，其中{control['per_class']['prompt_injection']['fp']}条被误标为注入。开发集上25/25注入捕获不能证明正常请求不被误伤。",
        f"- {last_turn['n']}条真实带历史的追问，仅传末轮时准确率{_percent(last_turn['raw_accuracy'])}；先接现有消解器后为{_percent(resolved['raw_accuracy'])}。这只是离线实验，尚未修改产品入口。",
        "- 注入标签不等于已阻断：/ask仍将该标签交给orchestrator，实际安全结果还取决于后续guard和数据库权限。", "",
    ]
    for name, dataset in report["datasets"].items():
        lines += [f"## {name}（{dataset['n']}题）", "", dataset["note"], "",
                  "标签数量：" + json.dumps(dataset["label_counts"], ensure_ascii=False), "",
                  "| 条件 | 五类准确率 | 四类路由一致率 | 诊断P/R | 闲聊P/R | 注入P/R | 注入误报 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for variant in report["variants"]:
            score = dataset["variants"][variant["name"]]
            classes = score["per_class"]
            lines.append(f"| {variant['name']} | {_percent(score['raw_accuracy'])} | {_percent(score['intent_route_accuracy'])} | {_pr(classes['diagnostic_anomaly'])} | {_pr(classes['chat_general'])} | {_pr(classes['prompt_injection'])} | {classes['prompt_injection']['fp']} |")
        lines += ["", "### 逐项消融（百分点，相对各自完整基线）", "",
                  "| 条件 | 基线 | 五类变化 | 改善题数 | 退化题数 |",
                  "|---|---|---:|---:|---:|"]
        for variant, delta in dataset["ablations"].items():
            lines.append(f"| {variant} | {delta['baseline']} | {delta['raw_delta_pp']:+.1f} | {len(delta['improved'])} | {len(delta['regressed'])} |")
        lines += ["", "### 完整级联的所有残余错误", ""]
        errors = [row for row in dataset["variants"]["cascade_full"]["rows"] if row["gold"] != row["pred"]]
        if not errors:
            lines.append("无（仅限本数据集）。")
        for row in errors:
            lines.append(f"- `{row['id']}` [{row['tag']}] `{row['gold']}` → `{row['pred']}`：{row['question']}")
            if row["input_question"] != row["question"]:
                lines.append(f"  - 消解后输入：{row['input_question']}")
            lines.append(f"  - 判定来源：`{row['reason']}`")
        lines.append("")
    lines += ["## 消融条件定义", ""]
    lines += [f"- `{variant['name']}`：{variant['description']}。" for variant in report["variants"]]
    lines += ["", "## 结论边界", ""]
    lines += [f"- {note}" for note in report["limitations"]]
    lines += ["", "## 方法风险检查（11/11）", "",
              "| 检查项 | 处理 |", "|---|---|",
              "| 辛普森悖论 | 开发集、负对照、上下文分开评分，保留逐标签结果。 |",
              "| 生态谬误 | 不从合成题总体均值推断所有真实用户。 |",
              "| 选择偏差 | 对抗题刻意选取，不能外推生产流量。 |",
              "| 碰撞变量偏差 | 未按预测结果筛题或控制共同结果变量。 |",
              "| 基础率忽略 | 报告标签数量、误报数和P/R。 |",
              "| 均值回归 | 不以开发集调优后的分数证明泛化提升。 |",
              "| 幸存者偏差 | 全部题及失败项保留，未剔除误判。 |",
              "| 多重比较 | 全部消融输出；不进行显著性宣称。 |",
              "| 分析路径选择 | 明示开发集参与调参，冻结逻辑后编写额外验证集。 |",
              "| 相关与因果混淆 | 组件开关效应仅限定本实现/本题集，不外推系统因果能力。 |",
              "| 反向因果 | 标签先于新验证预测确定；不按预测修改gold。 |", "",
              "## 复现", "", "```bash",
              "PYTHONPATH=src python -m insight_agent.eval.intent_ablation --output docs/intent_ablation_report.json --markdown docs/intent_ablation_report.md",
              "```", "", "JSON保留所有逐题记录、消融配置、数据集和源代码SHA-256。文件已存在时命令拒绝覆盖，需使用新输出路径。", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON output path (existing files are not overwritten)")
    parser.add_argument("--markdown", type=Path, help="New Markdown output path")
    parser.add_argument("--validation", type=Path, help="Validation JSON, default is the bundled fixture")
    args = parser.parse_args(argv)
    for path in (args.output, args.markdown):
        if path is not None and path.exists():
            parser.error(f"Output already exists; choose a new path: {path}")
    report = run_experiment(args.validation)
    for name, dataset in report["datasets"].items():
        print(f"\n{name}: n={dataset['n']}")
        for variant in VARIANTS:
            score = dataset["variants"][variant.name]
            injection = score["per_class"]["prompt_injection"]
            print(f"{variant.name:32s} accuracy={_percent(score['raw_accuracy']):>6} route={_percent(score['intent_route_accuracy']):>6} injection_fp={injection['fp']}")
    print("\nReproducibility: exact match across two complete runs; remote calls disabled.")
    outputs = ((args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n"), (args.markdown, render_markdown(report)))
    for path, content in outputs:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
