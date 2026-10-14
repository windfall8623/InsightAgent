"""Skill frequency analysis over the real JD corpus.

The point of doing this in code rather than by eye is auditability: anyone can
re-run it and get the same table, and every count is traceable to the postings
in :mod:`insight_agent.eval.jd_corpus`.

Matching is case-insensitive substring matching, which for Chinese text is
adequate but not perfect — "Agent" also matches inside "Agentic", and a posting
that says "多智能体" is credited to ``多智能体`` but not to ``Agent``. Counts are
therefore **lower bounds**, never inflated. Every matched term is also written
out with the postings that triggered it so a reviewer can spot bad matches.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.eval.jd_corpus import Posting, corpus

# Canonical skill -> the surface forms that count as a hit.
SKILL_TERMS: dict[str, tuple[str, ...]] = {
    # Agent 范式
    "多智能体/多Agent": ("多智能体", "多Agent", "多 agent", "multi-agent", "多智能体协同"),
    "Agent 编排/Orchestration": ("编排", "orchestration", "工作流", "workflow", "调度"),
    "Plan/规划": ("规划", "planning", "任务调度", "任务状态"),
    "记忆/Memory": ("记忆", "memory"),
    "上下文工程/Context": ("上下文", "context"),
    "工具调用/Tool use": ("工具", "tool", "工具调用", "技能", "skill", "插件"),
    "长程任务/长任务": ("长任务", "长程", "long-horizon", "持久化"),
    "Harness": ("harness", "Harness"),
    "Agent loop": ("agent loop", "Agent loop"),
    # 检索
    "RAG": ("RAG", "rag", "检索", "召回"),
    "知识库/知识管理": ("知识库", "企业知识", "知识"),
    "知识图谱": ("知识图谱", "knowledge graph"),
    "Text2SQL/SQL生成": ("sql生成", "Text2SQL", "text2sql", "SQL 生成"),
    # 协议与框架
    "MCP": ("MCP",),
    "A2A": ("A2A",),
    "协议/供应链安全": ("供应链", "supply-chain"),
    # 评测与观测
    "评测/Evaluation": ("评测", "评估", "evaluation", "eval"),
    "可观测性/Observability": (
        "可观测",
        "监控",
        "observability",
        "tracing",
        "OTel",
        "OpenTelemetry",
        "OTLP",
        "日志平台",
    ),
    "指标/度量": ("指标", "度量", "metrics"),
    "回归检测": ("回归", "regression"),
    "A/B 与反馈闭环": ("AB", "A/B", "反馈闭环", "反馈数据", "迭代闭环", "灰度"),
    # 工程与交付
    "Python": ("Python", "python"),
    "Go/Java/Node": ("Go", "Java", "Node", "TypeScript", "C++", "Rust"),
    "前端": ("React", "Vue", "WebAssembly", "WebGPU"),
    "FastAPI/后端框架": ("后端", "API", "服务"),
    "CI/CD 与交付": ("CI/CD", "灰度发布", "容灾", "交付"),
    "分布式与高并发": ("分布式", "高并发", "高可用", "稳定性", "扩展性"),
    "容器与云": ("云", "容器", "Kubernetes"),
    # 安全
    "提示注入防御": ("提示注入", "prompt injection"),
    "沙箱": ("沙箱", "sandbox", "隔离"),
    "红队/对抗测试": ("红队", "对抗测试", "自动化对抗", "red team", "red-team"),
    "权限与审计": ("权限", "审计", "授权", "管控"),
    "凭证/密钥保护": ("凭证", "密钥", "credential"),
    # 数据与模型
    "数据质量与治理": ("数据质量", "数据治理", "数据标准", "归因"),
    "模型训练/后训练": ("后训练", "强化学习", "对齐", "训练", "微调"),
    "成本控制": ("成本", "token", "效率"),
    "多模态": ("多模态", "图像", "视频", "语音"),
    "业务落地": ("业务", "落地", "场景"),
    "产品规划": ("产品", "规划", "商业化"),
}

# Skills the project must demonstrate, per the original requirement set.
REQUIRED_SKILLS = (
    "多智能体/多Agent",
    "Agent 编排/Orchestration",
    "记忆/Memory",
    "工具调用/Tool use",
    "MCP",
    "RAG",
    "评测/Evaluation",
    "可观测性/Observability",
    "提示注入防御",
    "沙箱",
    "Python",
    "CI/CD 与交付",
)


@dataclass(slots=True)
class SkillStat:
    skill: str
    postings: int
    share: float
    examples: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "postings": self.postings,
            "share": round(self.share, 4),
            "examples": self.examples[:3],
        }


def analyse(postings: list[Posting] | None = None) -> dict[str, Any]:
    """Compute the frequency table over the JD corpus."""

    # `postings or corpus()` would silently fall back to the full corpus
    # when handed an explicitly empty list.
    items = corpus() if postings is None else postings
    total = len(items)
    stats: list[SkillStat] = []

    for skill, terms in SKILL_TERMS.items():
        matched: list[Posting] = []
        for posting in items:
            # Title and scope are both JD text; a skill named in either counts.
            haystack = f"{posting.title} {posting.scope}".lower()
            if any(term.lower() in haystack for term in terms):
                matched.append(posting)
        stats.append(
            SkillStat(
                skill=skill,
                postings=len(matched),
                share=len(matched) / total if total else 0.0,
                examples=[p.title for p in matched[:3]],
            )
        )

    stats.sort(key=lambda s: (-s.postings, s.skill))

    coverage = {
        skill: next(s.postings for s in stats if s.skill == skill) for skill in REQUIRED_SKILLS
    }

    return {
        "total_postings": total,
        "skills": [s.to_payload() for s in stats],
        "required_skill_frequency": coverage,
        "distinct_skills_found": sum(1 for s in stats if s.postings > 0),
    }


def business_group_breakdown(postings: list[Posting] | None = None) -> dict[str, int]:
    items = corpus() if postings is None else postings
    counts: dict[str, int] = defaultdict(int)
    for p in items:
        counts[p.business_group] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def experience_breakdown(postings: list[Posting] | None = None) -> dict[str, int]:
    items = corpus() if postings is None else postings
    counts: dict[str, int] = defaultdict(int)
    for p in items:
        counts[p.work_year] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    report = analyse()
    print(
        f"corpus: {report['total_postings']} real postings "
        f"(Tencent official careers API, retrieved 2026-10-04)"
    )
    print(f"distinct skills matched: {report['distinct_skills_found']}/{len(SKILL_TERMS)}\n")

    print(f"{'skill':<26}{'postings':>9}{'share':>9}")
    print("-" * 44)
    for stat in report["skills"]:
        bar = "#" * int(stat["share"] * 40)
        print(f"{stat['skill']:<26}{stat['postings']:>9}{stat['share']:>8.0%}  {bar}")

    out = Path("eval_results/jd_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                **report,
                "business_groups": business_group_breakdown(),
                "experience": experience_breakdown(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
