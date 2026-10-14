"""JD corpus retrieved from Tencent's official careers API.

Provenance
----------
Source: ``careers.tencent.com/tencentcareer/api/post/Query`` (public, unauthenticated).
Queries run: ``keyword=Agent``, ``keyword=大模型应用``, ``keyword=RAG`` (pageSize 25, zh-cn).
Retrieved: 2026-10-04.

Two honesty caveats that must travel with any statistic computed from this file:

1. **These are 社招 (experienced-hire) postings, not 校招.** The campus portal
   (``join.qq.com``) is a POST endpoint that a GET-only fetcher cannot reach, and
   ``recruitType=2`` on the social endpoint did not filter (identical Count: 221).
   Senior postings state *more specific* technical requirements, so the skill
   distribution is richer than a campus corpus would be — but it is not the
   campus distribution, and must not be presented as one.
2. **Responsibility/requirement text is summarised, not verbatim.** Titles,
   cities, business groups and work-year fields are verbatim from the API.
   The prose was condensed by the fetching tool to respect length limits.

Every entry below is a real posting that existed in the API response. None are
reconstructed or invented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Posting:
    title: str
    city: str
    business_group: str
    work_year: str
    scope: str  # the responsibility/requirement summary as retrieved
    family: str  # which API query it came from


POSTINGS: list[Posting] = [
    # ---------------- keyword=Agent ----------------
    Posting(
        "Agent Security Expert",
        "新加坡",
        "CSIG",
        "未注明",
        "企业级Agent防护架构、运行时防护、提示注入防御、工具授权、凭证保护、"
        "供应链防御、自动化对抗测试；安全/系统工程背景，Go/Python/Rust/C++ 至少两种",
        "Agent",
    ),
    Posting(
        "Coding Agent Research Intern 108097",
        "新加坡",
        "IEG",
        "未注明",
        "编码Agent研究、自动游戏生成、调试优化、实验与论文；研究生在读，Python与深度学习基础，"
        "熟悉Agent/LLM",
        "Agent",
    ),
    Posting(
        "Agent Harness 工程师",
        "上海",
        "CSIG",
        "三年以上",
        "Agent loop、工具与技能抽象、上下文管理、长任务持久化、并发子Agent、沙箱、权限、"
        "tracing、监控、评测、调试工具",
        "Agent",
    ),
    Posting(
        "Agent安全专家",
        "深圳",
        "CSIG",
        "八年以上",
        "Agent安全产品架构；提示注入、越权工具调用、行为漂移、凭证泄露的运行时防护与沙箱隔离；"
        "MCP与工具供应链安全；自动化红队测试",
        "Agent",
    ),
    Posting(
        "Agent开发工程师",
        "深圳",
        "IEG",
        "五年以上",
        "端到端交付Agent产品：企业知识、检索、记忆、工具、MCP、多智能体协作、平台接入、"
        "评测、可观测性、CI/CD、测试、灰度发布、容灾、权限、审计、隔离",
        "Agent",
    ),
    Posting(
        "QQ-Agent产品经理",
        "深圳",
        "PCG",
        "一年以上",
        "Agent产品规划、通知与好友状态感知、检索/工具/记忆优化、行为反馈闭环",
        "Agent",
    ),
    Posting(
        "应用效能技术部-Agent开发工程师-Agent平台方向",
        "深圳",
        "PCG",
        "不限",
        "可扩展的Agent开发与运行时平台：编排、模型/工具接入、知识与上下文管理、评测、监控、"
        "平台化、高并发工程",
        "Agent",
    ),
    Posting(
        "WorkBuddy-Agent Harness 研发工程师",
        "深圳",
        "CSIG",
        "不限",
        "Harness 自进化、上下文工程、分布式多智能体协同、长期记忆、长程决策；"
        "自进化工具链与多维评测数据集",
        "Agent",
    ),
    Posting(
        "WorkBuddy-Agent Harness 算法工程师",
        "深圳",
        "CSIG",
        "不限",
        "自进化、上下文工程、多智能体协同、持久化记忆、长程决策；自进化SDK/平台与评测基础设施",
        "Agent",
    ),
    Posting(
        "微信搜索-Agent算法专家",
        "北京",
        "WXG",
        "一年以上",
        "搜索与深度研究Agent、后训练与强化学习、上下文与记忆、Agentic 强化学习",
        "Agent",
    ),
    Posting(
        "Agent Infra高级研发工程师",
        "深圳",
        "TEG",
        "两年以上",
        "强化学习与Agent的大规模沙箱与框架、自动化评测、实验复现；并发、稳定性、执行流、基础设施",
        "Agent",
    ),
    Posting(
        "WorkBuddy-Agent Harness 策略产品经理",
        "深圳",
        "CSIG",
        "不限",
        "Agent策略优化、上下文流转、任务执行；质量/可靠性/token效率平衡；评测标准与数据集；"
        "反馈与线上证据驱动",
        "Agent",
    ),
    Posting(
        "腾讯云DataBuddy-Agent研发专家",
        "深圳",
        "CSIG",
        "五年以上",
        "Agent运行时、可观测性、评测、上下文工程、记忆、自进化、Graph基座、知识挖掘；"
        "生成式AI与大数据工作流集成",
        "Agent",
    ),
    Posting(
        "Agent可观测研发工程师",
        "北京",
        "CSIG",
        "五年以上",
        "云日志平台与Agent可观测：采集、标准、SDK、OTLP/Kafka接入、OpenTelemetry生成式AI规范",
        "Agent",
    ),
    Posting(
        "腾讯视频-Agent 技术负责人",
        "北京",
        "PCG",
        "五年以上",
        "Agent架构与交付：知识、模型、规划、记忆、评测、治理、多模态、多智能体交互",
        "Agent",
    ),
    Posting(
        "混元AI Agent Harness Engineer",
        "北京",
        "TEG",
        "两年以上",
        "端到端tracing与监控、自动化评测、实验、回归检测、Agent调试工具、标注与评测平台、数据链路",
        "Agent",
    ),
    Posting(
        "混元Agent数据策略分析岗",
        "北京",
        "TEG",
        "一年以上",
        "分析开发者需求、用户反馈、对话、任务序列、工具使用、失败点；看板与归因；数据/模型/产品优先级",
        "Agent",
    ),
    Posting(
        "产品管理Agent开发架构师",
        "深圳",
        "CSIG",
        "五年以上",
        "识别可自动化场景；用技能、多智能体、工具、知识库、管控、评测、监控设计与上线Agent",
        "Agent",
    ),
    Posting(
        "轻量云OpenClaw AI Agent 高级工程师",
        "深圳",
        "CSIG",
        "三年以上",
        "Agent系统设计与优化、部署排障、prompt、工具、工作流、技能、检索式Agent、知识接入、"
        "可复用模型组件、文档与交付",
        "Agent",
    ),
    Posting(
        "AI产品经理-AI平台（Agent）方向",
        "深圳",
        "CDG",
        "两年以上",
        "Agent开发、监控、tracing、防护、调优、成本控制；多角色协作；指标/反馈/case治理服务质量",
        "Agent",
    ),
    Posting(
        "微信搜索-电商agent算法专家",
        "深圳",
        "WXG",
        "不限",
        "电商搜索Agent：意图理解、规划、检索、推荐",
        "Agent",
    ),
    # ---------------- keyword=大模型应用 ----------------
    Posting(
        "WeTalk-大模型应用研发工程师-(Agent Harness)",
        "北京",
        "TTS",
        "两年以上",
        "Agent loop与编排基础设施；上下文压缩、技能、记忆、检索、任务状态、评测、tracing与优化",
        "大模型应用",
    ),
    Posting(
        "微信基础-大模型应用开发工程师",
        "北京",
        "WXG",
        "不限",
        "大规模AI应用后端架构；Agent系统、长上下文对话、多模态流水线、可靠性、延迟与成本控制",
        "大模型应用",
    ),
    Posting(
        "英雄联盟手游-应用研究-大模型应用",
        "深圳",
        "IEG",
        "一年以上",
        "游戏内强化学习与语言模型Agent；多模态UGC/内容生成",
        "大模型应用",
    ),
    Posting(
        "腾讯元宝-大模型应用平台产品经理",
        "北京",
        "TEG",
        "三年以上",
        "应用平台建设、模型能力业务化、可复用组件",
        "大模型应用",
    ),
    Posting(
        "混元大模型应用算法工程师",
        "北京",
        "TEG",
        "两年以上",
        "后训练、Agent构建、对话与翻译改进、业务落地、模型增强与前沿研究",
        "大模型应用",
    ),
    Posting(
        "混元-大模型应用开发工程师",
        "深圳",
        "TEG",
        "三年以上",
        "跨平台移动AI应用；对话、翻译、写作、语音、文档；流式、上下文、路由、重试、端侧推理、质量工程",
        "大模型应用",
    ),
    Posting(
        "微信搜索-高级研究员-大模型应用搜索方向",
        "广州",
        "WXG",
        "三年以上",
        "生成式搜索排序、多模态理解、用户意图建模、后训练、反馈数据闭环、评测、检索、推理、Agentic搜索",
        "大模型应用",
    ),
    Posting(
        "微信小程序-大模型应用开发工程师-智能体方向",
        "广州",
        "WXG",
        "一年以上",
        "小程序智能体、行为理解、生成式界面、可观测、复杂任务调度、前沿研究与业务验证",
        "大模型应用",
    ),
    Posting(
        "企业微信-AI应用后台开发工程师-AI大模型应用",
        "广州",
        "WXG",
        "一年以上",
        "AI后台服务、RAG、任务自动化、模型调用调度、分布式系统、容灾、高可用",
        "大模型应用",
    ),
    Posting(
        "智能体-法律行业产品专家",
        "深圳",
        "PCG",
        "五年以上",
        "法律Agent能力、合规与合同流程、行业数据治理、客户调研、交付、监管科技合作",
        "大模型应用",
    ),
    Posting(
        "光子 AI-AI应用开发工程师",
        "深圳",
        "IEG",
        "三年以上",
        "游戏AI后台、Agent与内容生成模块、模型接入、编排、记忆、工具、部署、监控、成本优化",
        "大模型应用",
    ),
    Posting(
        "微信基础-后端开发工程师-Agent",
        "北京",
        "WXG",
        "两年以上",
        "产品Agent系统、架构与工作流、推理效率、生态接入、插件、工具、模型前沿能力落地",
        "大模型应用",
    ),
    # Reachable from both the 大模型应用 and RAG queries; kept once, with the
    # richer description, so title-uniqueness invariants hold.
    Posting(
        "腾讯云-AI前线部署工程师",
        "深圳",
        "CSIG",
        "五年以上",
        "客户AI方案交付、连接器与Skills定制、私有知识RAG、prompt/上下文工程、Skills、Agent、MCP集成、"
        "多智能体协同、业务结果度量、系统集成与指标",
        "大模型应用+RAG",
    ),
    Posting(
        "混元大语言数据质量工程师",
        "深圳",
        "TEG",
        "两年以上",
        "后训练数据质量体系、审核与归因链路、专家数据标注、质量标准、生产反馈",
        "大模型应用",
    ),
    Posting(
        "微信视频号-AIGC后台工程师",
        "广州",
        "WXG",
        "两年以上",
        "AIGC后台开发、图像/视频模型训练基础设施、服务性能、模型调度、稳定性、AI效能工具",
        "大模型应用",
    ),
    Posting(
        "微信小微-大模型算法研究员-Harness合版与记忆方向",
        "北京",
        "WXG",
        "不限",
        "Agent harness设计、任务调度、执行上下文、后训练数据沉淀、记忆生命周期、"
        "跨会话持久化、云端协同",
        "大模型应用",
    ),
    Posting(
        "微信支付-全栈开发工程师",
        "深圳",
        "CDG",
        "一年以上",
        "金融系统稳定性、基于AI的风险识别与修复、质量工具、代码评审、测试、容量规划、混沌演练、运维自动化",
        "大模型应用",
    ),
    Posting(
        "混元大语言数据工程师",
        "深圳",
        "TEG",
        "一年以上",
        "端到端文档数据处理、存储计算、解析、去重、质量筛选、Agent化数据生产、多模态数据",
        "大模型应用",
    ),
    Posting(
        "微信安全-风控算法高级工程师",
        "广州",
        "WXG",
        "一年以上",
        "欺诈与异常检测算法、大规模风控模型、业务协同、模型监控、系统架构、实时处理",
        "大模型应用",
    ),
    Posting(
        "智能体应用技术算法工程师",
        "北京",
        "TEG",
        "三年以上",
        "Agent平台、工具使用、记忆、多智能体协同、动态规划、数据分析、检索、SQL生成、知识图谱、"
        "评测、可观测性与优化",
        "大模型应用",
    ),
    Posting(
        "人力资源平台部-AI应用开发工程师",
        "深圳",
        "S3",
        "三年以上",
        "HR场景LLM与Agent项目、AI能力与组件标准化、方案设计与效果优化",
        "大模型应用",
    ),
    # ---------------- keyword=RAG ----------------
    Posting(
        "RAG技术负责人/架构师",
        "深圳",
        "TEG",
        "五年以上",
        "大规模RAG平台端到端架构、技术选型与落地；性能、召回、准确率、迭代与技术方向",
        "RAG",
    ),
    Posting(
        "腾讯乐享-高级产品策划",
        "深圳",
        "CSIG",
        "五年以上",
        "企业知识协作产品AI能力、Agentic知识库需求、评测与迭代机制",
        "RAG",
    ),
    Posting(
        "腾讯云-大模型产品解决方案架构师",
        "深圳",
        "CSIG",
        "三年以上",
        "客户LLM场景理解、完整方案设计、测试与迁移、交付风险管理",
        "RAG",
    ),
    Posting(
        "AI infra规划架构师",
        "深圳",
        "TEG",
        "五年以上",
        "公司级AI平台与基础设施规划、训练与推理效率、异构芯片、推理优化、分布式训练",
        "RAG",
    ),
    Posting(
        "腾讯游戏-AI agent开发-数据产品方向",
        "深圳",
        "IEG",
        "不限",
        "游戏业务数据产品与平台、数据采集处理分析可视化、工作流集成、AI+数据研究",
        "RAG",
    ),
    Posting(
        "腾讯营销-数据Agent产品经理",
        "深圳",
        "CDG",
        "一年以上",
        "营销数据Agent产品、评测与迭代闭环、数据工具与诊断Skills封装",
        "RAG",
    ),
    Posting(
        "测试工程师",
        "深圳",
        "TEG",
        "一年以上",
        "安全系统测试、用例与缺陷分析；支持Agent协同与评测、生产事故复盘、质量报告、测试方法改进",
        "RAG",
    ),
    Posting(
        "保险科技-大模型高级工程师",
        "深圳",
        "CDG",
        "三年以上",
        "保险域模型后训练、指令数据、对齐、奖励建模、Agentic-RL环境；准确性/合规/推理/满意度自动化评测",
        "RAG",
    ),
    Posting(
        "企业微信-AI应用后台开发工程师-业务挖掘和应用",
        "广州",
        "WXG",
        "一年以上",
        "企业知识抽取与业务流程建模、NLP与知识图谱、智能应用、分布式高可用后台",
        "RAG",
    ),
    Posting(
        "浏览器内核与自动化工程师（AI Agent领域）",
        "深圳",
        "TEG",
        "三年以上",
        "为Agent提供高效语义忠实的网页上下文；Playwright/Chromium定制、渲染完整性、跨平台稳定性、"
        "指纹模拟、内容清洗与抽取、查询式摘要",
        "RAG",
    ),
    Posting(
        "智能体套件-高级产品经理-CodeBuddy/WorkBuddy",
        "深圳",
        "CSIG",
        "五年以上",
        "Agent套件路线图、模块组合、计费、权限、商业化",
        "RAG",
    ),
    Posting(
        "游戏Agentic Engineer",
        "深圳",
        "IEG",
        "三年以上",
        "为合作游戏工作室落地AI：识别并重构工作流、可复用实践、智能NPC与Data Agent",
        "RAG",
    ),
    Posting(
        "腾讯AI协作工具-AI Agent研发工程师",
        "深圳",
        "CSIG",
        "不限",
        "编排、工具与Skill、对话、上下文、规划、编辑器集成；RAG、评测、稳定性、成本、MCP、"
        "端侧推理、WebAssembly/WebGPU、多模态",
        "RAG",
    ),
    Posting(
        "金融科技-AI数据产品经理-数据开发方向",
        "深圳",
        "CDG",
        "三年以上",
        "金融风控合规数据产品；AI用于需求/编码/报表/洞察/知识接入/交易分析；数据标准与治理",
        "RAG",
    ),
    Posting(
        "腾讯云-TBDS大数据产品商业化负责人",
        "深圳",
        "CSIG",
        "五年以上",
        "商业化、定价、封装、市场策略、大客户、Data+AI能力",
        "RAG",
    ),
    Posting(
        "代号：Craft-高级工程师-Agentic Engineer",
        "深圳",
        "IEG",
        "三年以上",
        "轻量游戏创作Agent、上下文处理与工作流、生成内容质量、Agentic改造、可复用方法与工具",
        "RAG",
    ),
    Posting(
        "ima知识号-搜推策略产品经理",
        "北京",
        "CSIG",
        "两年以上",
        "知识频道搜索与推荐策略、内容质量评估、分类标签、策略评估",
        "RAG",
    ),
    # ---------------- 第二批（Agent 页2 + 智能体关键词）仅研发/测试岗 ----------------
    Posting(
        "腾讯金融-AI Agent 研发高级工程师",
        "深圳",
        "CDG",
        "五年以上",
        "智能体核心平台、插件与工具框架；完善评测、诊断与可观测体系；性能、稳定性与安全性",
        "Agent",
    ),
    Posting(
        "腾讯投资-高级开发工程师-AI Agent",
        "北京",
        "CDG",
        "三年以上",
        "智能体架构、对话管理、任务规划、记忆、外部工具调用；企业级RAG；接入业务系统并保障可用性与安全",
        "Agent",
    ),
    Posting(
        "高级投放 Agent 研发工程师",
        "北京",
        "CDG",
        "三年以上",
        "广告投放智能体的运行时、编排、记忆、技能体系；复杂任务可靠性与决策质量；评测标准",
        "Agent",
    ),
    Posting(
        "企业微信-AI Agent Harness 全栈工程师",
        "广州",
        "WXG",
        "五年以上",
        "AgentLoop、工具编排与运行环境；上下文、长期记忆、任务恢复；开发轨迹回放等工具；"
        "评测、沙箱和可观测基础设施",
        "Agent",
    ),
    Posting(
        "微信-大模型(Agent)应用开发后台工程师",
        "北京",
        "WXG",
        "不限",
        "高并发智能体后端、对话与多模态交互；复杂规划与多轮决策；技能、多智能体、微调、推理优化与成本控制",
        "Agent",
    ),
    Posting(
        "混元Agent评测Infra工程师",
        "北京",
        "TEG",
        "三年以上",
        "沙箱、依赖管理、网络与并发调度等评测底座；保证结果可复现、可诊断；评测指标",
        "Agent",
    ),
    Posting(
        "混元Agent评测Infra工程专家",
        "北京",
        "TEG",
        "三年以上",
        "统一接入多类智能体评测基准；沙箱、依赖、网络、调度；保障评分可信与问题归因",
        "Agent",
    ),
    Posting(
        "微信小程序-AI Agent 开发工程师-小工具",
        "广州",
        "WXG",
        "两年以上",
        "支持多智能体的分布式框架；沙箱、云环境、消息链路与存储；性能与数据一致性；分布式训练",
        "Agent",
    ),
    Posting(
        "微信基础-大模型算法工程师-Agent方向",
        "北京",
        "WXG",
        "不限",
        "微信垂类智能体；上下文、任务拆解、多工具能力；后训练提升业务指标",
        "Agent",
    ),
    Posting(
        "微信小程序-搜索算法工程师-Agent方向",
        "广州",
        "WXG",
        "两年以上",
        "面向智能体的搜索全链路；查询改写、向量表示、重排模型；评估与训练数据闭环",
        "Agent",
    ),
    Posting(
        "生态发展部-资深游戏测试工程师-Agent评测",
        "深圳",
        "IEG",
        "三年以上",
        "智能体全链路评测与质量标准；分级指标与基准测试集；自动评测与模型交叉验证；缺陷定位",
        "Agent",
    ),
    Posting(
        "混元Agent强化学习框架工程师",
        "北京",
        "TEG",
        "一年以上",
        "智能体与强化学习框架；统一模型、工具、任务、上下文、日志与评估组件；并发与一致性",
        "Agent",
    ),
    Posting(
        "智能体- Agent Harness 全栈工程师",
        "深圳",
        "CSIG",
        "五年以上",
        "Agent Harness基础设施端到端；开发者工具、上下文与记忆体系、"
        "多智能体协同、运行时优化、可观测性",
        "智能体",
    ),
    Posting(
        "AI智能体评测高级工程师-WorkBuddy",
        "深圳",
        "CSIG",
        "五年以上",
        "LLM与Agent评测体系、基准、行业指标、测试集、执行器、标准与质量流程",
        "智能体",
    ),
    Posting(
        "AI智能体评测高级工程师",
        "北京",
        "CSIG",
        "五年以上",
        "LLM与Agent质量评测框架、基准、评估流程、数据集与执行工具",
        "智能体",
    ),
    Posting(
        "智能体-全栈测试工程师（WorkBuddy方向）",
        "深圳",
        "CSIG",
        "三年以上",
        "Agent测试与评测能力：白盒、端到端、自动化与专项测试；"
        "Prompt、RAG、Tool Use、风险检测与质量指标",
        "智能体",
    ),
    Posting(
        "企业智能体-全栈开发工程师-WorkBuddy",
        "深圳",
        "CSIG",
        "三年以上",
        "企业级Managed Agents、多租户架构、模型接口、质量与性能治理、"
        "可复用AI开发Harness、标准化交付",
        "智能体",
    ),
    Posting(
        "智能体-数据治理高级工程师-WorkBuddy",
        "深圳",
        "CSIG",
        "五年以上",
        "业务、事件、数据与指标模型设计；埋点与数仓标准；数据质量；支撑管道、分析、评测与模型优化",
        "智能体",
    ),
    Posting(
        "智能体-全栈开发工程师-WorkBuddy",
        "武汉",
        "CSIG",
        "五年以上",
        "协作产品架构、AI交互与编排、跨端工程、微服务、CI/CD、可观测性",
        "智能体",
    ),
]


def corpus() -> list[Posting]:
    return list(POSTINGS)


def corpus_size() -> int:
    return len(POSTINGS)
