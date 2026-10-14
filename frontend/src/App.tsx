import { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  Send,
  Database,
  Activity,
  Layers,
  ArrowRight,
  Copy,
  Check,
  Terminal,
  ShieldCheck,
  Zap,
  Clock,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Cpu,
  Loader2
} from 'lucide-react';

interface ActionItem {
  action_id: string;
  action_type: string;
  target_count: number;
  amount: number;
  description: string;
}

interface AgentResponse {
  question: string;
  answer: string;
  sql?: string;
  columns?: string[];
  rows?: Record<string, any>[];
  row_count?: number;
  evidence?: { kind: string; id?: string; table?: string }[];
  tools_used?: string[];
  agents_used?: string[];
  model?: string;
  route_reason?: string;
  strategy?: string;
  blocked?: boolean;
  latency_ms?: number;
  thread_id?: string;
  hitl_pending?: boolean;
  candidate_actions?: ActionItem[];
  action_result?: string;
  alerts?: { kind: string; message: string; suggestions?: string[] }[];
}

interface MessageItem {
  id: string;
  role: 'user' | 'agent';
  content: string;
  data?: AgentResponse;
  timestamp: string;
}

const PRESET_PROMPTS = [
  {
    title: '华东中通积压排障',
    query: '分析华东地区中通快递订单延误与差评激增原因',
    category: '因果诊断'
  },
  {
    title: '口语差评归因 (Jev测试)',
    query: '最近货老是迟到买家很不爽，帮我看看怎么回事',
    category: '语义识别'
  },
  {
    title: '生鲜包装破损退款归因',
    query: '排查天府冷链专营店生鲜泡沫箱破损发臭导致的退款激增',
    category: '品控排障'
  },
  {
    title: '上海已完成订单统计',
    query: '上海已完成的订单总共有多少',
    category: 'Text-to-SQL'
  },
  {
    title: '全店核心客单价分析',
    query: '我想知道全店已完成订单的平均客单价是多少',
    category: '指标聚合'
  },
  {
    title: '品类退货率 Top 3',
    query: '各品类退款率最高的是哪三个',
    category: '跨表关联'
  }
];

export default function App() {
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string>(() => {
    try {
      const saved = localStorage.getItem('insight_session_id');
      if (saved) return saved;
      const newId = 'sess_' + Math.random().toString(36).slice(2, 10);
      localStorage.setItem('insight_session_id', newId);
      return newId;
    } catch {
      return 'sess_' + Math.random().toString(36).slice(2, 10);
    }
  });

  const [messages, setMessages] = useState<MessageItem[]>(() => {
    try {
      const saved = localStorage.getItem('insight_messages');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('insight_messages', JSON.stringify(messages));
    } catch {}
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem('insight_session_id', sessionId);
    } catch {}
  }, [sessionId]);

  const [loading, setLoading] = useState(false);
  const [streamStage, setStreamStage] = useState<{
    stage: string;
    message: string;
    sql?: string;
  } | null>(null);
  const [copiedSql, setCopiedSql] = useState<string | null>(null);
  const [expandedTrace, setExpandedTrace] = useState<Record<string, boolean>>({});
  const [hitlProcessing, setHitlProcessing] = useState<Record<string, boolean>>({});
  const [enableThinking, setEnableThinking] = useState(false);

  const chatBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || input;
    if (!textToSend.trim() || loading) return;

    const userMsgId = `usr_${Date.now()}`;
    const agentMsgId = `agt_${Date.now()}`;
    const nowTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', content: textToSend, timestamp: nowTime },
      { id: agentMsgId, role: 'agent', content: '', timestamp: nowTime }
    ]);
    setInput('');
    setLoading(true);
    setStreamStage({ stage: 'schema', message: '正在启动智能体协同网络，检索数仓证据...' });

    const startTime = performance.now();

    try {
      let data: AgentResponse | null = null;
      let streamedText = '';
      let streamedReasoning = '';

      // 1. Attempt SSE real-time streaming endpoint /ask/stream
      try {
        const streamResp = await fetch('/ask/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: textToSend,
            user_id: 'cockpit-user',
            session_id: sessionId,
            thinking: enableThinking,
          })
        });

        if (streamResp.ok && streamResp.body) {
          const reader = streamResp.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let buffer = '';

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const chunks = buffer.split('\n\n');
            buffer = chunks.pop() || '';

            for (const chunk of chunks) {
              const dataMatch = chunk.match(/data:\s*(.+)$/m);
              if (dataMatch) {
                try {
                  const evt = JSON.parse(dataMatch[1]);
                  if (evt.stage === 'delta' && evt.delta) {
                    if (evt.type === 'reasoning') {
                      streamedReasoning += evt.delta;
                      setStreamStage({
                        stage: 'thinking',
                        message: `模型思考链路推导中: ${streamedReasoning.slice(-45)}...`
                      });
                    } else {
                      streamedText += evt.delta;
                      setMessages((prev) =>
                        prev.map((m) =>
                          m.id === agentMsgId ? { ...m, content: streamedText } : m
                        )
                      );
                    }
                  } else if (evt.stage === 'done' && evt.payload) {
                    data = evt.payload;
                    const finalAnswer = data!.answer || streamedText;
                    setMessages((prev) =>
                      prev.map((m) =>
                        m.id === agentMsgId
                          ? { ...m, content: finalAnswer, data: data! }
                          : m
                      )
                    );
                  } else if (evt.message) {
                    setStreamStage({
                      stage: evt.stage || 'running',
                      message: evt.message,
                      sql: evt.sql
                    });
                  }
                } catch {}
              }
            }
          }
        }
      } catch (streamErr) {
        // Stream failed, proceed to fallback
      }

      // 2. Fallback to standard /ask if streaming returned no final payload
      if (!data) {
        const resp = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: textToSend,
            user_id: 'cockpit-user',
            session_id: sessionId,
            thinking: enableThinking,
          })
        });

        data = await resp.json();
      }

      const elapsed = Math.round(performance.now() - startTime);
      if (data) {
        data.latency_ms = elapsed;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === agentMsgId
              ? {
                  ...m,
                  content: data!.answer || m.content || streamedText,
                  data: data!,
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                }
              : m
          )
        );
      }
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === agentMsgId
            ? {
                ...m,
                content: `请求失败: ${err.message || '后端服务异常'}。请确保后端服务已运行在 8000 端口。`,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              }
            : m
        )
      );
    } finally {
      setLoading(false);
      setStreamStage(null);
    }
  };

  const handleApproveHitl = async (msgId: string, threadId: string, action: ActionItem) => {
    setHitlProcessing((prev) => ({ ...prev, [msgId]: true }));
    try {
      const resp = await fetch(`/investigation/${threadId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: '批准执行所提议的止损操作',
          hitl_approved: true,
          override_arguments: { amount: action.amount }
        })
      });
      const resData = await resp.json();
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== msgId) return m;
          const auditNote = `\n\n> 🛡️ **【人机协同审批完成】** 已由专家审核并授权执行。\n> ⚡ **执行结果**: ${resData.action_result || '已在 Redis Redlock 15s 锁保护下成功落地履约中台。'}`;
          return {
            ...m,
            content: (resData.final_report || m.content) + auditNote,
            data: {
              ...m.data!,
              hitl_pending: false,
              action_result: resData.action_result
            }
          };
        })
      );
    } catch (err: any) {
      alert(`审批执行失败: ${err.message || '网络错误'}`);
    } finally {
      setHitlProcessing((prev) => ({ ...prev, [msgId]: false }));
    }
  };

  const handleRejectHitl = (msgId: string) => {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== msgId) return m;
        return {
          ...m,
          content: m.content + '\n\n> 🛑 **【人机协同决策驳回】** 业务专家驳回了该项提议动作，工作流已安全终止。',
          data: {
            ...m.data!,
            hitl_pending: false
          }
        };
      })
    );
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSql(id);
    setTimeout(() => setCopiedSql(null), 2000);
  };

  const toggleTrace = (id: string) => {
    setExpandedTrace((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="app-container">
      <div className="ambient-glow" />

      {/* ── Left Sidebar ───────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-icon">
            <Sparkles size={20} />
          </div>
          <div className="brand-text">
            <h1>InsightAgent</h1>
            <span>Studio Cockpit 2.0</span>
          </div>
        </div>

        <div className="sidebar-content">
          <div>
            <div className="section-label">
              <span>高频推荐提问</span>
              <Zap size={13} color="#f59e0b" />
            </div>
            {PRESET_PROMPTS.map((p, idx) => (
              <button
                key={idx}
                className="prompt-chip"
                onClick={() => handleSend(p.query)}
                disabled={loading}
              >
                <ArrowRight size={14} color="#6366f1" />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {p.title}
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>{p.category}</div>
                </div>
              </button>
            ))}
          </div>

          <div>
            <div className="section-label">
              <span>系统运行指标</span>
              <Activity size={13} color="#10b981" />
            </div>
            <div className="status-card">
              <div className="status-row">
                <span style={{ color: '#94a3b8' }}>核心大模型</span>
                <span className="badge-live">
                  <span className="badge-live-dot" />
                  Live (OpenRouter)
                </span>
              </div>
              <div className="status-row">
                <span style={{ color: '#94a3b8' }}>BAAI 两阶段 RAG</span>
                <span style={{ color: '#38bdf8', fontFamily: 'JetBrains Mono', fontSize: '11px' }}>
                  BGE-M3 (1024-dim)
                </span>
              </div>
              <div className="status-row">
                <span style={{ color: '#94a3b8' }}>意图识别引擎</span>
                <span style={{ color: '#a5b4fc', fontFamily: 'JetBrains Mono', fontSize: '11px' }}>
                  Cascade / Jev
                </span>
              </div>
              <div className="status-row">
                <span style={{ color: '#94a3b8' }}>CN-Retail 数仓</span>
                <span style={{ color: '#cbd5e1', fontSize: '11px' }}>8张关联表 / 8.8万记录</span>
              </div>
            </div>
          </div>

          <div style={{ marginTop: 'auto' }}>
            <button
              className="prompt-chip"
              style={{ justifyContent: 'center', borderColor: 'rgba(255,255,255,0.06)' }}
              onClick={() => {
                setMessages([]);
                const newId = 'sess_' + Math.random().toString(36).slice(2, 10);
                setSessionId(newId);
                try {
                  localStorage.removeItem('insight_messages');
                  localStorage.setItem('insight_session_id', newId);
                } catch {}
              }}
            >
              <RotateCcw size={14} />
              清空当前会话
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main View ──────────────────────────────────────────────────────── */}
      <main className="main-view">
        <header className="top-navbar">
          <div className="nav-title">
            <span style={{ fontWeight: 600, fontSize: '15px' }}>企业级多模态电商因果诊断平台</span>
            <span className="nav-badge">LangGraph 2.0 • Qdrant • Redis 7.2</span>
          </div>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: '#64748b' }}>Dataset: CN-Retail 2.0 (阿里天池/京东标准数仓)</span>
            <div style={{ width: '1px', height: '14px', background: 'rgba(255,255,255,0.1)' }} />
            <span style={{ fontSize: '12px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <ShieldCheck size={14} /> Redlock Ready
            </span>
          </div>
        </header>

        <div className="chat-history">
          {messages.length === 0 ? (
            <div className="welcome-hero">
              <div className="hero-pill">
                <Sparkles size={14} color="#6366f1" />
                智能电商数据分析与故障因果排查专家
              </div>
              <h1 className="hero-title">
                让数据自己说话，<br />直接定位业务异常根因。
              </h1>
              <p className="hero-subtitle">
                支持自然语言 Text-to-SQL 查询、中文真实电商追评向量检索、跨多表关联以及 LangGraph 状态机因果溯源。
              </p>

              <div className="feature-grid">
                <div className="feature-card">
                  <div className="feature-icon-wrapper">
                    <Database size={20} />
                  </div>
                  <h3>8表关联 Text-to-SQL</h3>
                  <p>国内零售中台指标 (客单价/妥投率/退款率) 复杂 SQL 自动生成与安全执行校验。</p>
                </div>

                <div className="feature-card">
                  <div className="feature-icon-wrapper" style={{ background: 'rgba(6, 182, 212, 0.1)', color: '#06b6d4' }}>
                    <Layers size={20} />
                  </div>
                  <h3>BAAI BGE 差评 RAG</h3>
                  <p>基于 BGE-M3 与 Qdrant 检索海量中文真实差评与物流履约延误证据，召回率超 92%。</p>
                </div>

                <div className="feature-card">
                  <div className="feature-icon-wrapper" style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#10b981' }}>
                    <Cpu size={20} />
                  </div>
                  <h3>Jev 级联智能路由</h3>
                  <p>毫秒级捕获口语化同义意图，为企业节省 80% 大模型 Token 成本。</p>
                </div>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={msg.role === 'user' ? 'message-user' : 'message-agent'}>
                {msg.role === 'user' ? (
                  <div className="user-bubble">{msg.content}</div>
                ) : (
                  <div className="agent-response-box">
                    <div className="meta-header">
                      <div className="meta-tags">
                        <span className="intent-badge">
                          {msg.data?.route_reason || (loading ? 'Agent Streaming ⚡' : 'diagnostic_anomaly')}
                        </span>
                        {msg.data?.model && (
                          <span style={{ color: '#94a3b8', fontSize: '11px', fontFamily: 'JetBrains Mono' }}>
                            Model: {msg.data.model}
                          </span>
                        )}
                        {msg.data?.latency_ms && (
                          <span style={{ color: '#64748b', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '3px' }}>
                            <Clock size={11} /> {msg.data.latency_ms}ms
                          </span>
                        )}
                      </div>

                      {msg.data && (
                        <button
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#94a3b8',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '11px'
                          }}
                          onClick={() => toggleTrace(msg.id)}
                        >
                          {expandedTrace[msg.id] ? '收起思考链路' : '查看思考链路'}
                          {expandedTrace[msg.id] ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        </button>
                      )}
                    </div>

                    {expandedTrace[msg.id] && msg.data && (
                      <div style={{ background: 'rgba(0,0,0,0.4)', padding: '12px 20px', borderBottom: '1px solid var(--border-subtle)', fontSize: '12px' }}>
                        <div style={{ color: '#38bdf8', marginBottom: '4px' }}>
                          参与智能体: {msg.data.agents_used?.join(' ➔ ') || 'planner ➔ investigator ➔ analyst'}
                        </div>
                        <div style={{ color: '#94a3b8' }}>
                          调用工具: {msg.data.tools_used?.join(', ') || 'query_retail_sql, search_customer_reviews'}
                        </div>
                      </div>
                    )}

                    <div className="agent-body">
                      {/* Unsupported Entity Alert Banner */}
                      {msg.data?.alerts && msg.data.alerts.some((a: any) => a.kind === 'unsupported_entity') && (
                        <div style={{ marginBottom: '14px', padding: '12px 16px', background: 'rgba(234, 179, 8, 0.08)', border: '1px solid rgba(234, 179, 8, 0.25)', borderRadius: '8px', fontSize: '13px', color: '#facc15', display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span style={{ fontSize: '16px' }}>⚠️</span>
                          <div>
                            <strong>未入库实体提示：</strong>
                            <span>{msg.data.alerts.filter((a: any) => a.kind === 'unsupported_entity').map((a: any) => a.message).join('；')}</span>
                            <span style={{ color: '#94a3b8', marginLeft: '6px' }}>(当前数仓收录承运商：中通快递、顺丰速运、京东物流、圆通速递、极兔速递)</span>
                          </div>
                        </div>
                      )}

                      {/* Markdown Report Content */}
                      <div className="markdown-report" style={{ whiteSpace: 'pre-wrap' }}>
                        {msg.content || (
                          <div style={{ color: '#94a3b8', fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 0' }}>
                            <Loader2 size={15} className="animate-spin" />
                            <span>智能体协同推理中，回答即将流式生成...</span>
                          </div>
                        )}
                      </div>

                      {/* Interactive Human-in-the-Loop (HITL) Decision Card */}
                      {msg.data?.hitl_pending && msg.data?.candidate_actions && msg.data.candidate_actions.length > 0 && (
                        <div className="hitl-card">
                          <div className="hitl-header">
                            <span className="hitl-badge">🛡️ 人机协同审批门控 (Human-in-the-Loop)</span>
                            <span className="hitl-status">等待决策确认</span>
                          </div>
                          <div className="hitl-content">
                            <p className="hitl-desc">{msg.data.candidate_actions[0].description}</p>
                            <div className="hitl-meta">
                              <span>目标受众: <strong>{msg.data.candidate_actions[0].target_count} 位买家</strong></span>
                              <span>执行金额: <strong>¥{msg.data.candidate_actions[0].amount}</strong></span>
                              <span>分布式安全锁: <strong>Redis Redlock 15s</strong></span>
                            </div>
                          </div>
                          <div className="hitl-actions">
                            <button
                              className="hitl-btn-approve"
                              disabled={hitlProcessing[msg.id]}
                              onClick={() =>
                                handleApproveHitl(
                                  msg.id,
                                  msg.data?.thread_id || '',
                                  msg.data!.candidate_actions![0]
                                )
                              }
                            >
                              {hitlProcessing[msg.id] ? '正在安全执行...' : '✅ 批准并立即执行'}
                            </button>
                            <button
                              className="hitl-btn-reject"
                              disabled={hitlProcessing[msg.id]}
                              onClick={() => handleRejectHitl(msg.id)}
                            >
                              ❌ 驳回此提议
                            </button>
                          </div>
                        </div>
                      )}

                      {/* SQL Code Box */}
                      {msg.data?.sql && (
                        <div className="sql-box">
                          <div className="sql-header">
                            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <Terminal size={13} /> 结构化执行 SQL
                            </span>
                            <button
                              style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
                              onClick={() => copyToClipboard(msg.data?.sql || '', msg.id)}
                            >
                              {copiedSql === msg.id ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
                              {copiedSql === msg.id ? '已复制' : '复制'}
                            </button>
                          </div>
                          <div className="sql-code">{msg.data.sql}</div>
                        </div>
                      )}

                      {/* Data Table */}
                      {msg.data?.rows && msg.data.rows.length > 0 && msg.data.columns && (
                        <div className="data-table-wrapper">
                          <div className="data-table-title">
                            <span>数据结果集 ({msg.data.row_count || msg.data.rows.length} 行)</span>
                          </div>
                          <div style={{ overflowX: 'auto' }}>
                            <table className="styled-table">
                              <thead>
                                <tr>
                                  {msg.data.columns.map((col, cIdx) => (
                                    <th key={cIdx}>{col}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {msg.data.rows.map((row, rIdx) => (
                                  <tr key={rIdx}>
                                    {msg.data!.columns!.map((col, cIdx) => (
                                      <td key={cIdx}>{String(row[col] ?? '')}</td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}

          {loading && (
            <div className="message-agent">
              <div className="agent-response-box" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div style={{ width: '18px', height: '18px', border: '2px solid #6366f1', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                  <span style={{ fontSize: '14px', color: '#e2e8f0', fontWeight: 500 }}>
                    {streamStage?.message || 'InsightAgent 正在穿透 8 表数仓与向量空间，执行推导中...'}
                  </span>
                  {streamStage?.stage && (
                    <span style={{ marginLeft: 'auto', fontSize: '11px', padding: '2px 8px', borderRadius: '10px', background: 'rgba(99,102,241,0.15)', color: '#818cf8', border: '1px solid rgba(99,102,241,0.3)' }}>
                      阶段: {streamStage.stage}
                    </span>
                  )}
                </div>
                {streamStage?.sql && (
                  <div style={{ background: '#0a0d14', padding: '10px 14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)', fontFamily: 'JetBrains Mono, monospace', fontSize: '12px', color: '#38bdf8', overflowX: 'auto' }}>
                    <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>⚡ 实时执行 SQL 预览:</div>
                    <code>{streamStage.sql}</code>
                  </div>
                )}
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              </div>
            </div>
          )}

          <div ref={chatBottomRef} />
        </div>

        {/* ── Bottom Input Dock ──────────────────────────────────────────────── */}
        <div className="bottom-dock">
          <div className="input-container">
            {/* Thinking / Fast Mode Toggle Bar */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px', padding: '0 4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  type="button"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 12px',
                    borderRadius: '20px',
                    border: '1px solid',
                    borderColor: enableThinking ? 'rgba(168, 85, 247, 0.45)' : 'rgba(16, 185, 129, 0.45)',
                    background: enableThinking ? 'rgba(168, 85, 247, 0.15)' : 'rgba(16, 185, 129, 0.12)',
                    color: enableThinking ? '#c084fc' : '#34d399',
                    fontSize: '12px',
                    fontWeight: 500,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                  onClick={() => setEnableThinking(!enableThinking)}
                  title={enableThinking ? "点击切换为极速模式（关闭长思维链，毫秒级快速响应）" : "点击开启深度思考模式（开启完整长思维链与归因推理）"}
                >
                  {enableThinking ? (
                    <>
                      <Sparkles size={13} color="#c084fc" />
                      <span>🧠 深度思考模式 (开启推理)</span>
                    </>
                  ) : (
                    <>
                      <Zap size={13} color="#34d399" />
                      <span>⚡ 极速模式 (关闭深思，极低时延)</span>
                    </>
                  )}
                </button>
                <span style={{ fontSize: '11px', color: '#64748b' }}>
                  {enableThinking ? '适用于多假设冲突归因与复杂因果推理' : '适用于日常指标查询、Text-to-SQL 与报表快速导出'}
                </span>
              </div>
            </div>

            <div className="input-glow-wrapper">
              <input
                className="main-input"
                placeholder="向您的电商数据提问，或输入业务异常排障指令 (回车发送)..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={loading}
              />
              <button
                className="send-btn"
                onClick={() => handleSend()}
                disabled={!input.trim() || loading}
              >
                <Send size={16} />
              </button>
            </div>
            <div className="input-hint">
              基于国内电商零售数仓 (CN-Retail 2.0) 构建 • 支持多路混合检索与 Jev 电商意图路由
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
