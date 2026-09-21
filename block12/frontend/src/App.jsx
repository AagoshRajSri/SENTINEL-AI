import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import FilamentNav from './FilamentNav';
import intentsImg from './assets/intents.jpg';
import policyImg from './assets/policy.jpg';
import metricsImg from './assets/metrics.jpg';
import docsImg from './assets/docs.jpg';

import LightPillar from './LightPillar';

const SUGGESTIONS = {
  complaints: [
    { text: 'My package was supposed to arrive three days ago and tracking has not updated.', type: 'complaint' },
    { text: 'The driver left a note saying delivered but nothing is at my door.', type: 'complaint' },
    { text: 'I was charged twice for the same order and need a refund immediately.', type: 'complaint' },
    { text: 'The product arrived shattered. I need a replacement sent today.', type: 'complaint' },
    { text: 'Your driver was rude and left the package in the rain. This is unacceptable.', type: 'complaint' },
  ],
  compliments: [
    { text: 'Your support team resolved my issue in under 10 minutes. Outstanding.', type: 'compliment' },
    { text: 'The replacement arrived next day. I am genuinely impressed.', type: 'compliment' },
    { text: 'The agent was patient and thorough. Please pass on my thanks.', type: 'compliment' },
    { text: 'I really appreciate the quick follow-up and the clear communication.', type: 'compliment' },
    { text: 'The support representative went above and beyond to help me out. Great service!', type: 'compliment' },
  ],
};

const INTENT_META = {
  'Delivery Issues':                 { dot: '--dot-delivery' },
  'Order Issues':                    { dot: '--dot-order' },
  'Product Issues':                  { dot: '--dot-product' },
  'Payment & Refunds':               { dot: '--dot-payment' },
  'Account & Access':                { dot: '--dot-account' },
  'Subscription & Digital Services': { dot: '--dot-subscription' },
  'Severe Escalation':               { dot: '--dot-severe' },
};

function App() {
  const [query, setQuery] = useState('');
  const [lastQuery, setLastQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [stats, setStats] = useState(null);
  
  const [copied, setCopied] = useState(false);
  const [toastMsg, setToastMsg] = useState('');
  
  const [popoverOpen, setPopoverOpen] = useState(false);
  
  const inputRef = useRef(null);

  useEffect(() => {
    fetch('/api/stats')
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) setStats(data);
      })
      .catch(() => {});
  }, []);

  const handleInput = (e) => {
    setQuery(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px';
  };

  const runTriage = async (textToRun = query) => {
    const text = textToRun.trim();
    if (!text) {
      inputRef.current?.focus();
      return;
    }
    
    setLastQuery(text);
    setLoading(true);
    setError('');
    setResults(null);

    try {
      const res = await fetch('/api/triage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Server error ' + res.status);
      }
      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runTriage();
    }
  };

  const handleCopy = () => {
    if (!results?.draft_reply) return;
    navigator.clipboard.writeText(results.draft_reply)
      .then(() => {
        setCopied(true);
        showToast('Reply copied to clipboard');
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => showToast('Copy failed — select and copy manually'));
  };

  const showToast = (msg) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(''), 2000);
  };
  
  const f1Display = stats?.macro_f1 != null ? `Macro-F1 ${stats.macro_f1.toFixed(3)}` : 'Benchmark';

  return (
    <>
      <div className="shape-waves-bg">
        
      </div>

      <div className="app-content">
        <FilamentNav />

        <main>

        <div style={{ width: '100%', height: '100vh', position: 'fixed', top: 0, left: 0, zIndex: 0, pointerEvents: 'none' }}>
          <LightPillar
            topColor="#6366F1"
            bottomColor="#7200de"
            intensity={1}
            rotationSpeed={0.3}
            glowAmount={0.002}
            pillarWidth={3}
            pillarHeight={0.4}
            noiseIntensity={0.5}
            pillarRotation={25}
            interactive={false}
            mixBlendMode="screen"
            quality="high"
          />
        </div>

          <div className="container">
      <section id="triage" className="page-section">
        
            
            <div className="layout-top">
            <div className="layout-left">
              <section className="hero anim-2" aria-labelledby="hero-heading">
              <h1 id="hero-heading">Instantly understand<br/>customer issues</h1>
              <p>Paste a customer message below. Sentinel analyzes the intent, scores the escalation risk, and drafts a tailored response based on your support guidelines.</p>
            </section>

            <div className="input-console anim-3">
              <textarea
                id="query-input"
                ref={inputRef}
                value={query}
                onChange={handleInput}
                onKeyDown={handleKeyDown}
                placeholder="Paste a customer message..."
                rows="1"
                aria-label="Customer message"
                autoComplete="off"
                spellCheck="true"
                style={{ fieldSizing: 'content' }}
              />
              <button 
                  className="submit-btn" 
                  aria-label={results ? "Clear triage" : "Run triage"} 
                  type="button"
                  onClick={() => {
                    if (results) {
                      setQuery("");
                      setResults(null);
                    } else {
                      runTriage();
                    }
                  }}
                  disabled={loading}
                >
                {results ? (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M4 4l8 8M12 4l-8 8"/>
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M8 12V4M4 8l4-4 4 4"/>
                  </svg>
                )}
              </button>
              <div className="input-hint"><kbd>Enter</kbd> run &middot; <kbd>Shift+Enter</kbd> newline</div>
            </div>
            </div>
            <div className="layout-right">
              <div className="chips-section anim-4">
              <div className="chips-col">
                <div className="chips-label">Complaints</div>
                <div className="chips-row" role="group" aria-label="Complaint examples">
                  {SUGGESTIONS.complaints.map((s, i) => (
                    <button 
                      key={i} 
                      className="chip" 
                      type="button" 
                      title={s.text}
                      onClick={() => {
                        setQuery(s.text);
                        runTriage(s.text);
                      }}
                    >
                      <span className={`chip-dot ${s.type}`} aria-hidden="true"></span>
                      <span>{s.text.length > 52 ? s.text.slice(0,52)+'...' : s.text}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="chips-col">
                <div className="chips-label">Compliments</div>
                <div className="chips-row" role="group" aria-label="Compliment examples">
                  {SUGGESTIONS.compliments.map((s, i) => (
                    <button 
                      key={i} 
                      className="chip" 
                      type="button" 
                      title={s.text}
                      onClick={() => {
                        setQuery(s.text);
                        runTriage(s.text);
                      }}
                    >
                      <span className={`chip-dot ${s.type}`} aria-hidden="true"></span>
                      <span>{s.text.length > 52 ? s.text.slice(0,52)+'...' : s.text}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            </div></div>
              {(loading || error || results) && (
              <div className="results-container" style={{ display: 'block' }} aria-live="polite">
                {loading && <div className="progress-bar active"></div>}
                {loading && <div className="loading-status active">Classifying...</div>}

                {results && !loading && (
                  <div className="results-body" style={{ display: 'grid' }}>
                    
                    {/* Drafted reply */}
                    <div className="result-block reply result-enter visible">
                        <div className="liquid-border"></div>
                        <div className="liquid-border-inner"></div>
                      <div className="block-label">
                        <span>Drafted reply</span>
                        <button className={`copy-btn ${copied ? 'copied' : ''}`} aria-label="Copy reply to clipboard" type="button" onClick={handleCopy}>
                          {copied ? (
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                              <path d="M2 6.5l3 3 5.5-5.5"/>
                            </svg>
                          ) : (
                            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                              <rect x="3.5" y="3.5" width="7" height="7" rx="1"/>
                              <path d="M1.5 8.5V1.5h7"/>
                            </svg>
                          )}
                          <span>{copied ? 'Copied' : 'Copy'}</span>
                        </button>
                      </div>
                      <div className="reply-text">{results.draft_reply || '—'}</div>
                    </div>

                    {/* Classification */}
                    <div className="result-block intent-block result-enter visible" style={{ transitionDelay: '40ms' }}>
                      <div className="block-label"><span>Classification</span></div>

                      <div className="intent-row">
                        <span className="intent-dot" aria-hidden="true" style={{ background: `var(${INTENT_META[results.intent]?.dot || '--dot-delivery'})` }}></span>
                        <span className="intent-label">{results.intent || '—'}</span>
                      </div>

                      <div className="conf-row">
                        <div className="conf-meta">
                          <span>Confidence</span>
                          <span className="conf-value">{(results.confidence || 0).toFixed(2)}</span>
                        </div>
                        <div className="conf-track">
                          <div className="conf-fill" style={{ transform: `scaleX(${results.confidence || 0})` }}></div>
                        </div>
                      </div>

                      <div className="esc-row">
                        <span className="esc-key">Escalation</span>
                        <span className="esc-dot" aria-hidden="true" style={{ background: results.should_escalate ? 'var(--danger)' : 'var(--ok)' }}></span>
                        <span className="esc-val" style={{ color: results.should_escalate ? 'var(--danger)' : 'var(--text-2)' }}>
                          {results.should_escalate ? 'Required' : 'Not required'} (score {results.escalation_score})
                        </span>
                      </div>
                    </div>

                    {/* Reasoning + Policy */}
                    <div className="result-block result-enter visible" style={{ transitionDelay: '80ms' }}>
                      <div className="detail-section">
                        <div className="block-label"><span>Reasoning</span></div>
                        <div className="detail-text">{results.reasoning || '—'}</div>
                      </div>
                      <div className="detail-section">
                        <div className="block-label"><span>Policy adherence</span></div>
                        <div className="detail-text">{results.policy_adherence || '—'}</div>
                      </div>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="error-line visible">
                    <span>{error}</span>
                    <button className="retry-btn" type="button" onClick={() => runTriage(lastQuery)}>Retry</button>
                  </div>
                )}
              </div>
            )}
            </section>
                  <section id="intents" className="page-section detail-block">
        <div className="detail-content">
          <div className="detail-dot" style={{ backgroundColor: 'var(--ok)' }}></div>
          <h2>Semantic Routing Engine</h2>
          <p>Sentinel-AI reads between the lines so your team doesn't have to. It maps every frantic customer message into one of seven core buckets with ruthless accuracy. By analyzing semantic intent rather than just keyword matching, it catches nuances like sarcasm and extreme frustration.</p>
          <p>Anything sketchy or below the 0.65 confidence threshold gets instantly flagged for human eyes. Zero hallucinations, zero rogue behavior. Just surgical categorization.</p>
        </div>
        <div className="detail-image-wrapper">
          <img src={intentsImg} alt="Intents Graphic" className="detail-image" />
        </div>
      </section>
      
      <section id="policy" className="page-section detail-block">
        <div className="detail-image-wrapper">
          <img src={policyImg} alt="Policy Graphic" className="detail-image" />
        </div>
        <div className="detail-content">
          <div className="detail-dot" style={{ backgroundColor: 'var(--danger)' }}></div>
          <h2>The Policy Guardrail</h2>
          <p>Our AI plays by the rules—your rules. Every drafted reply runs through a secondary verification layer (the "Policy Guard") which intercepts wild promises and rogue refunds before they ever see an outbound queue.</p>
          <p>If a draft attempts to issue a refund exceeding your defined limits without manager authorization, the system blocks the draft and routes it to the escalation queue. Think of it as a relentless bouncer for your support emails.</p>
        </div>
      </section>
      
      <section id="metrics" className="page-section detail-block">
        <div className="detail-content">
          <div className="detail-dot" style={{ backgroundColor: '#6366F1' }}></div>
          <h2>Performance Telemetry</h2>
          <p>Fast, accurate, and completely transparent. At 410ms per inference, it triages faster than you can blink. We're consistently clocking a 94.2% accuracy rate with a 0.2% false-positive rate for Severe Escalations.</p>
          <p>Every decision is logged, scored, and mapped on our real-time telemetry dashboard. Because guessing is for amateurs, and your customer support deserves precision engineering.</p>
        </div>
        <div className="detail-image-wrapper">
          <img src={metricsImg} alt="Metrics Graphic" className="detail-image" />
        </div>
      </section>
      
      <section id="docs" className="page-section detail-block">
        <div className="detail-image-wrapper">
          <img src={docsImg} alt="Docs Graphic" className="detail-image" />
        </div>
        <div className="detail-content">
          <div className="detail-dot" style={{ backgroundColor: '#FF9FFC' }}></div>
          <h2>API & Integration</h2>
          <p>Plug and play with strictly typed JSON payloads. No PhD required. Hook our REST API up to your existing helpdesk and watch the magic happen.</p>
          <p>Our endpoints are aggressively rate-limited (in a good way) and incredibly resilient. Full SDKs are available for Node.js, Python, and Go, complete with automatic retry logic and exponential backoff. Build on top of Sentinel-AI in under an hour.</p>
        </div>
      </section>
    </div>
        </main>

        <div className={`toast ${toastMsg ? 'show' : ''}`} role="status" aria-live="polite">
          {toastMsg}
        </div>
      </div>
    </>
  );
}

export default App;
