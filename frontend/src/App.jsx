import { useEffect, useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const SAMPLES = ['奥智的使命是什么？', '矫直切割机最大线速度是多少？', '复绕机由哪些系统组成？']

const fmt = (value, digits = 3) => value == null ? '—' : Number(value).toFixed(digits)
const ms = (value) => value == null ? '—' : `${Math.round(value)} ms`

function Logo() {
  return <div className="logo-mark" aria-label="奥智智库"><span>O</span><i /></div>
}

function StatusPill({ health, error }) {
  const online = health?.status === 'ok'
  return (
    <div className={`status-pill ${online ? 'online' : error ? 'offline' : ''}`}>
      <span className="status-dot" />
      {online ? `知识库就绪 · ${health.chunks} 块` : error ? '后端未连接' : '正在连接'}
    </div>
  )
}

function CandidateCard({ item, stage }) {
  const score = stage === 'vector' ? item.vector_score
    : stage === 'bm25' ? item.bm25_score
      : stage === 'rrf' ? item.rrf_score : item.reranker_score
  return (
    <details className="candidate-card">
      <summary>
        <span className="rank">{item.rank}</span>
        <span className="candidate-main">
          <strong>{item.file}</strong>
          <small>{item.section}</small>
        </span>
        <span className="score">{fmt(score, stage === 'rrf' ? 6 : 4)}</span>
      </summary>
      <div className="candidate-body">
        <p>{item.preview}</p>
        <div className="score-tags">
          {item.vector_rank && <span>向量 #{item.vector_rank} · {fmt(item.vector_score, 4)}</span>}
          {item.bm25_rank && <span>BM25 #{item.bm25_rank} · {fmt(item.bm25_score, 4)}</span>}
          {item.rrf_score != null && <span>RRF · {fmt(item.rrf_score, 6)}</span>}
          {item.reranker_score != null && <span>精排 · {fmt(item.reranker_score, 4)}</span>}
        </div>
        <code>{item.chunk_id}</code>
      </div>
    </details>
  )
}

function Stage({ number, eyebrow, title, subtitle, timing, children, accent = 'blue' }) {
  return (
    <section className={`stage stage-${accent}`}>
      <header className="stage-header">
        <span className="stage-number">{number}</span>
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        {timing && <span className="timing">{timing}</span>}
      </header>
      {children}
    </section>
  )
}

function CitationAnswer({ text = '' }) {
  const parts = useMemo(() => text.split(/(\[资料\d+\])/g), [text])
  return <div className="answer-text">{parts.map((part, i) => /^\[资料\d+\]$/.test(part)
    ? <span className="citation" key={i}>{part}</span>
    : <span key={i}>{part}</span>)}</div>
}

function PipelineSkeleton() {
  return <div className="loading-panel">
    <div className="loader-orbit"><span /><span /><span /></div>
    <h2>正在穿过检索链路</h2>
    <p>向量召回与关键词召回并行执行，随后融合、精排并生成答案。</p>
    <div className="loading-steps"><i /><i /><i /><i /></div>
  </div>
}

function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(false)
  const [question, setQuestion] = useState('奥智的使命是什么？')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/health`)
      if (!response.ok) throw new Error('后端未就绪')
      setHealth(await response.json())
      setHealthError(false)
    } catch {
      setHealthError(true)
    }
  }

  useEffect(() => { checkHealth() }, [])

  const submit = async (event) => {
    event?.preventDefault()
    if (!question.trim() || loading) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || '问答请求失败')
      setResult(payload)
    } catch (err) {
      setError(err.message || '无法连接问答服务')
      checkHealth()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><Logo /><div><strong>奥智智库</strong><span>RAG KNOWLEDGE LAB</span></div></div>
        <nav><a href="#ask">智能问答</a><a href="#pipeline">检索链路</a></nav>
        <StatusPill health={health} error={healthError} />
      </header>

      <main>
        <section className="hero" id="ask">
          <div className="hero-copy">
            <span className="hero-label">ENTERPRISE KNOWLEDGE · TRACEABLE AI</span>
            <h1>每一个答案，<br /><em>都看得见来路。</em></h1>
            <p>面向公司制度、设备标准与培训知识的可追溯问答。检索、融合、重排、生成，全链路透明呈现。</p>
          </div>
          <form className="ask-box" onSubmit={submit}>
            <label htmlFor="question">向知识库提问</label>
            <div className="input-row">
              <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={500}
                placeholder="例如：奥智的使命是什么？" rows={3} />
              <button type="submit" disabled={loading || !question.trim()}>
                {loading ? '检索中' : '开始检索'}<span>↗</span>
              </button>
            </div>
            <div className="samples"><span>试试：</span>{SAMPLES.map((item) => <button type="button" key={item} onClick={() => setQuestion(item)}>{item}</button>)}</div>
          </form>
        </section>

        {error && <div className="error-banner"><strong>请求没有完成</strong><span>{error}</span><button onClick={checkHealth}>重新检测</button></div>}
        {loading && <PipelineSkeleton />}

        {result && <div className="results" id="pipeline">
          <section className="answer-panel">
            <div className="answer-kicker"><span>FINAL ANSWER</span><b>{ms(result.timings.total_ms)}</b></div>
            <h2>{result.question}</h2>
            <CitationAnswer text={result.answer} />
            <div className="source-strip">{result.sources.map((source) => <span key={source.chunk_id}>资料{source.index} · {source.file}</span>)}</div>
          </section>

          <div className="pipeline-heading">
            <div><span className="eyebrow">RETRIEVAL TRACE</span><h2>一次回答如何被找到</h2></div>
            <p>点击任意候选块，查看它在每一步的分数与来源。</p>
          </div>

          <Stage number="01" eyebrow="QUERY" title="原始问题" subtitle="保持用户原始语义，不做隐藏改写" accent="ink">
            <div className="query-chip">{result.question}</div>
          </Stage>

          <div className="parallel-stage">
            <Stage number="02A" eyebrow="DENSE RETRIEVAL" title="BGE-M3 向量召回" subtitle="语义相似度 Top 10" timing={ms(result.timings.vector_ms)} accent="blue">
              <div className="candidate-list">{result.trace.vector.map((item) => <CandidateCard key={item.chunk_id} item={item} stage="vector" />)}</div>
            </Stage>
            <div className="parallel-mark"><span>并行</span></div>
            <Stage number="02B" eyebrow="LEXICAL RETRIEVAL" title="BM25 关键词召回" subtitle="精确词项匹配 Top 10" timing={ms(result.timings.bm25_ms)} accent="orange">
              <div className="candidate-list">{result.trace.bm25.map((item) => <CandidateCard key={item.chunk_id} item={item} stage="bm25" />)}</div>
            </Stage>
          </div>

          <Stage number="03" eyebrow="WEIGHTED FUSION" title="加权 RRF 融合" subtitle="向量 0.6 · BM25 0.4 · 去重后保留前 10" accent="violet">
            <div className="formula"><code>score = 0.6 / (60 + vector_rank) + 0.4 / (60 + bm25_rank)</code></div>
            <div className="candidate-grid">{result.trace.rrf.map((item) => <CandidateCard key={item.chunk_id} item={item} stage="rrf" />)}</div>
          </Stage>

          <Stage number="04" eyebrow="CROSS ENCODER" title="Qwen3 精排" subtitle="逐一判断问题与候选证据的直接相关性" timing={ms(result.timings.rerank_ms)} accent="green">
            <div className="candidate-grid">{result.trace.reranked.map((item) => <CandidateCard key={item.chunk_id} item={item} stage="reranked" />)}</div>
          </Stage>

          <Stage number="05" eyebrow="GENERATION" title="DeepSeek 证据化生成" subtitle="仅使用精排前 5 个文本块回答并添加资料引用" timing={ms(result.timings.generation_ms)} accent="red">
            <div className="final-contexts">{result.sources.map((source) => <details key={source.chunk_id}><summary><span>资料{source.index}</span><strong>{source.file}</strong><small>{source.section}</small></summary><p>{source.content}</p></details>)}</div>
          </Stage>

          <section className="timing-panel">
            <div><span>向量检索</span><b>{ms(result.timings.vector_ms)}</b></div>
            <div><span>BM25</span><b>{ms(result.timings.bm25_ms)}</b></div>
            <div><span>Reranker</span><b>{ms(result.timings.rerank_ms)}</b></div>
            <div><span>DeepSeek</span><b>{ms(result.timings.generation_ms)}</b></div>
            <div className="total"><span>端到端</span><b>{ms(result.timings.total_ms)}</b></div>
          </section>
        </div>}
      </main>
      <footer><span>AOZHI INTELLIGENCE · INTERNAL KNOWLEDGE SYSTEM</span><span>375 chunks · 1024 dimensions · 3 domains</span></footer>
    </div>
  )
}

export default App
