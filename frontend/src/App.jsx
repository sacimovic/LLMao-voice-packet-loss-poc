import React, { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const clamp = (value, min, max) => Math.min(Math.max(value, min), max)

const formatSeconds = (ms) => `${(ms / 1000).toFixed(2)}s`

const tokenizeWords = (text) => {
  if (!text || !text.trim()) return []
  return text.trim().split(/\s+/).filter(Boolean)
}

const alignWordPairs = (sourceText, targetText) => {
  const sourceWords = tokenizeWords(sourceText)
  const targetWords = tokenizeWords(targetText)
  const n = sourceWords.length
  const m = targetWords.length
  if (!n && !m) return { pairs: [], degradedWords: sourceWords, repairedWords: targetWords }

  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0))
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      dp[i][j] = sourceWords[i] === targetWords[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }

  const pairs = []
  let i = 0
  let j = 0
  while (i < n || j < m) {
    const sourceWord = i < n ? sourceWords[i] : null
    const targetWord = j < m ? targetWords[j] : null
    if (sourceWord && targetWord) {
      if (sourceWord === targetWord) {
        pairs.push({ degraded: sourceWord, repaired: targetWord, match: true })
        i += 1
        j += 1
      } else {
        const nextSource = dp[i + 1]?.[j] ?? 0
        const nextTarget = dp[i]?.[j + 1] ?? 0
        if (nextSource === nextTarget) {
          pairs.push({ degraded: sourceWord, repaired: targetWord, match: false })
          i += 1
          j += 1
        } else if (nextSource > nextTarget) {
          pairs.push({ degraded: sourceWord, repaired: null, match: false })
          i += 1
        } else {
          pairs.push({ degraded: null, repaired: targetWord, match: false })
          j += 1
        }
      }
    } else if (sourceWord) {
      pairs.push({ degraded: sourceWord, repaired: null, match: false })
      i += 1
    } else if (targetWord) {
      pairs.push({ degraded: null, repaired: targetWord, match: false })
      j += 1
    }
  }

  return { pairs, degradedWords: sourceWords, repairedWords: targetWords }
}

export default function App() {
  const [file, setFile] = useState(null)
  const [degrade, setDegrade] = useState(30)
  const [degradeMode, setDegradeMode] = useState('percentage')
  const [windowMs, setWindowMs] = useState(40)
  const [windowStartMs, setWindowStartMs] = useState(0)
  const [audioDurationMs, setAudioDurationMs] = useState(null)
  const [loading, setLoading] = useState(false)
  const [asrText, setAsrText] = useState('')
  const [repairedText, setRepairedText] = useState('')
  const [originalUrl, setOriginalUrl] = useState(null)
  const [degradedUrl, setDegradedUrl] = useState(null)
  const [ttsUrl, setTtsUrl] = useState(null)
  const [combinedUrl, setCombinedUrl] = useState(null)
  const [error, setError] = useState('')
  const [statusMessage, setStatusMessage] = useState('Upload an audio file to get started.')
  const [statusPhase, setStatusPhase] = useState('idle')
  const [elapsedMs, setElapsedMs] = useState(null)

  const statusTimers = useRef([])
  const processingStartRef = useRef(null)
  const objectUrlsRef = useRef({ original: null, degraded: null, tts: null, combined: null })

  const maxWindowMs = audioDurationMs ? Math.max(40, Math.floor(audioDurationMs / 3)) : 400
  const windowStartMax = audioDurationMs ? Math.max(Math.floor(audioDurationMs - windowMs), 0) : 0
  const windowEndMs = windowStartMs + windowMs
  const clampedWindowEndMs = audioDurationMs ? Math.min(windowEndMs, audioDurationMs) : windowEndMs
  const windowStartPercent = audioDurationMs ? (windowStartMs / audioDurationMs) * 100 : 0
  const windowEndPercent = audioDurationMs ? (clampedWindowEndMs / audioDurationMs) * 100 : 0
  const windowLengthTrack = maxWindowMs ? (windowMs / maxWindowMs) * 100 : 0
  const windowStartTrack = windowStartMax ? (windowStartMs / windowStartMax) * 100 : 0

  const { pairs, degradedWords, repairedWords } = useMemo(
    () => alignWordPairs(asrText, repairedText),
    [asrText, repairedText]
  )

  const { stats: nerdStats, counts: diffCounts } = useMemo(() => {
    const counts = { matches: 0, wrong: 0, insertions: 0 }
    pairs.forEach((pair) => {
      if (pair.match) counts.matches += 1
      if (pair.degraded && !pair.match) counts.wrong += 1
      if (pair.repaired && !pair.match) counts.insertions += 1
    })

    const repairedCount = repairedWords.length
    const matchPct = repairedCount ? Math.round((counts.matches / repairedCount) * 100) : null
    const correctionCount = repairedCount ? Math.max(counts.wrong, counts.insertions) : null
    const approxWer = repairedCount && correctionCount !== null
      ? Math.round((correctionCount / Math.max(repairedCount, 1)) * 100)
      : null

    const lossDescriptor = degradeMode === 'window'
      ? audioDurationMs
        ? `Window ${windowMs} ms @ ${formatSeconds(windowStartMs)}`
        : `Window ${windowMs} ms`
      : `${degrade}% random drop`

    const stats = [
      { label: 'Loss strategy', value: lossDescriptor },
      {
        label: 'Clip length',
        value: audioDurationMs ? formatSeconds(audioDurationMs) : '—'
      },
      { label: 'Whisper words', value: degradedWords.length ? `${degradedWords.length}` : '—' },
      { label: 'Repaired words', value: repairedCount ? `${repairedCount}` : '—' },
      { label: 'Exact matches', value: repairedCount ? `${counts.matches} (${matchPct}% coverage)` : '—' },
      { label: 'Approx. WER', value: approxWer !== null ? `${approxWer}%` : '—' },
      {
        label: 'Corrections applied',
        value: correctionCount !== null ? `${correctionCount}` : '—'
      },
      {
        label: 'Processing time',
        value: elapsedMs ? `${(elapsedMs / 1000).toFixed(1)}s` : '—'
      }
    ]

    return { stats, counts }
  }, [pairs, degradeMode, degrade, windowMs, windowStartMs, audioDurationMs, degradedWords.length, repairedWords.length, elapsedMs])

  const flaggedTokenCount = Math.max(diffCounts.wrong, diffCounts.insertions)
  const correctedTokenCount = Math.max(diffCounts.insertions, diffCounts.wrong)

  const scheduleStatusTimeline = () => {
    statusTimers.current.forEach(clearTimeout)
    statusTimers.current = []
    const steps = [
      { delay: 0, message: 'Uploading audio & simulating packet loss…' },
      { delay: 1500, message: 'Running Whisper transcription on degraded signal…' },
      { delay: 3500, message: 'Repairing transcript with local FLAN-T5…' },
      { delay: 5500, message: 'Rebuilding speech with XTTS voice cloning…' }
    ]
    statusTimers.current = steps.map(({ delay, message }) => (
      setTimeout(() => setStatusMessage(message), delay)
    ))
  }

  useEffect(() => () => {
    statusTimers.current.forEach(clearTimeout)
    Object.values(objectUrlsRef.current).forEach((url) => url && URL.revokeObjectURL(url))
  }, [])

  useEffect(() => {
    if (!originalUrl) {
      setAudioDurationMs(null)
      return
    }
    const audio = new Audio()
    audio.src = originalUrl
    const onLoaded = () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        setAudioDurationMs(audio.duration * 1000)
      } else {
        setAudioDurationMs(null)
      }
    }
    audio.addEventListener('loadedmetadata', onLoaded)
    return () => {
      audio.removeEventListener('loadedmetadata', onLoaded)
      audio.src = ''
    }
  }, [originalUrl])

  useEffect(() => {
    if (!audioDurationMs) return
    const maxWindow = clamp(Math.floor(audioDurationMs / 3), 40, audioDurationMs)
    setWindowMs((prev) => clamp(prev, 40, maxWindow))
  }, [audioDurationMs])

  useEffect(() => {
    if (!audioDurationMs) return
    const maxStart = Math.max(Math.floor(audioDurationMs - windowMs), 0)
    setWindowStartMs((prev) => clamp(prev, 0, maxStart))
  }, [windowMs, audioDurationMs])

  const updateUrl = (setter, key, value) => {
    if (objectUrlsRef.current[key]) {
      URL.revokeObjectURL(objectUrlsRef.current[key])
    }
    objectUrlsRef.current[key] = value
    setter(value)
  }

  const handleFile = (e) => {
    const selected = e.target.files[0]
    setFile(selected)
    if (selected) {
      statusTimers.current.forEach(clearTimeout)
      statusTimers.current = []
      updateUrl(setOriginalUrl, 'original', URL.createObjectURL(selected))
      setStatusPhase('idle')
      setStatusMessage('Ready to process the selected audio.')
      setAsrText('')
      setRepairedText('')
      setElapsedMs(null)
      setAudioDurationMs(null)
      setWindowStartMs(0)
      setWindowMs(40)
      updateUrl(setDegradedUrl, 'degraded', null)
      updateUrl(setTtsUrl, 'tts', null)
      updateUrl(setCombinedUrl, 'combined', null)
    } else {
      statusTimers.current.forEach(clearTimeout)
      statusTimers.current = []
      updateUrl(setOriginalUrl, 'original', null)
      setAsrText('')
      setRepairedText('')
      setElapsedMs(null)
      setAudioDurationMs(null)
      setWindowStartMs(0)
      setWindowMs(40)
      setDegradeMode('percentage')
      updateUrl(setDegradedUrl, 'degraded', null)
      updateUrl(setTtsUrl, 'tts', null)
      updateUrl(setCombinedUrl, 'combined', null)
    }
  }

  const b64ToBlobUrl = (b64) => {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'audio/wav' })
    return URL.createObjectURL(blob)
  }

  const onProcess = async () => {
    if (!file) {
      setError('Pick an audio file first.')
      return
    }
    setError('')
    setLoading(true)
    setAsrText('')
    setRepairedText('')
    updateUrl(setDegradedUrl, 'degraded', null)
    updateUrl(setTtsUrl, 'tts', null)
    updateUrl(setCombinedUrl, 'combined', null)
    setStatusPhase('processing')
    setStatusMessage('Uploading audio & simulating packet loss…')
    setElapsedMs(null)
    processingStartRef.current = performance.now()
    scheduleStatusTimeline()

    const form = new FormData()
    form.append('file', file)
    form.append('degrade_mode', degradeMode)
    if (degradeMode === 'window') {
      form.append('degrade_percent', '0')
      form.append('window_ms', windowMs.toString())
      form.append('window_start_ms', windowStartMs.toString())
    } else {
      form.append('degrade_percent', degrade.toString())
      form.append('window_ms', windowMs.toString())
      form.append('window_start_ms', windowStartMs.toString())
    }
    form.append('whisper_model', 'base')
    form.append('repair_model', 'google/flan-t5-small')
    form.append('synth_all_text', 'true')

    try {
      const res = await fetch(`${API_URL}/process`, { method: 'POST', body: form })
      let data
      try {
        data = await res.json()
      } catch (jsonErr) {
        throw new Error(`Backend returned invalid JSON: ${jsonErr}`)
      }

      if (!res.ok) {
        const message = data?.error || `Backend error (${res.status})`
        throw new Error(message)
      }

      const elapsed = performance.now() - processingStartRef.current
      setElapsedMs(elapsed)
      setStatusPhase('done')
      setStatusMessage(`Processing complete in ${(elapsed / 1000).toFixed(1)}s. Enjoy the results!`)
      setAsrText(data.asr_text || '')
      setRepairedText(data.repaired_text || '')
      if (data.degraded_wav_b64) updateUrl(setDegradedUrl, 'degraded', b64ToBlobUrl(data.degraded_wav_b64))
      if (data.tts_wav_b64) updateUrl(setTtsUrl, 'tts', b64ToBlobUrl(data.tts_wav_b64))
      if (data.combined_wav_b64) updateUrl(setCombinedUrl, 'combined', b64ToBlobUrl(data.combined_wav_b64))
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      setError(message)
      setStatusPhase('error')
      setStatusMessage(message)
    } finally {
      statusTimers.current.forEach(clearTimeout)
      statusTimers.current = []
      setLoading(false)
    }
  }

  const renderWordColumn = (targetKey) => {
    if (!pairs.length) {
      return <p className="comparison-placeholder">Run a job to compare transcripts.</p>
    }

    return (
      <div className="word-stream">
        {pairs.map((pair, idx) => {
          const token = pair[targetKey]
          if (!token) {
            const missingClass = 'word-wrong word-missing'
            const missingTitle = targetKey === 'degraded' ? 'Inserted by repair model' : 'Removed during repair'
            return (
              <span
                key={`${targetKey}-${idx}-missing`}
                className={`word-token ${missingClass}`}
                title={missingTitle}
              >
                [missing]
              </span>
            )
          }

          const baseClasses = ['word-token']
          if (pair.match) {
            baseClasses.push('word-correct')
          } else if (targetKey === 'degraded') {
            baseClasses.push('word-wrong')
          } else {
            baseClasses.push('word-correct', 'word-correct-delta')
          }

          return (
            <span
              key={`${targetKey}-${idx}-${token}`}
              className={baseClasses.join(' ')}
              title={pair.match ? 'Matched between degraded and repaired' : targetKey === 'degraded' ? 'Requires repair' : 'Added by repair model'}
            >
              {token}
            </span>
          )
        })}
      </div>
    )
  }

  return (
    <div className="app-shell">
      <div className="aurora" aria-hidden="true" />
      <main className="main-card">
        <header className="header">
          <div>
            <p className="eyebrow">Hackathon proof of concept</p>
            <h1>LLMao · Voice Packet Loss Lab</h1>
            <p className="lede">Upload a clip, dial in how badly packets drop, then watch the stack degrade, transcribe, repair, and resynthesize your speech.</p>
          </div>
          <div className={`status-pill status-${statusPhase}`}>
            {statusPhase === 'processing' ? (
              <span className="spinner" aria-hidden="true" />
            ) : (
              <span className={`status-dot dot-${statusPhase}`} aria-hidden="true" />
            )}
            <span>{statusMessage}</span>
          </div>
        </header>

        <section className="controls">
          <label className="file-picker">
            <span className="label">Load audio</span>
            <div className="chip">{file ? file.name : 'Choose a file'} </div>
            <input type="file" accept="audio/*" onChange={handleFile} />
          </label>

          <div className="mode-toggle" role="group" aria-label="Degradation mode">
            <button
              type="button"
              className={`toggle-option ${degradeMode === 'percentage' ? 'is-active' : ''}`}
              onClick={() => setDegradeMode('percentage')}
              disabled={loading}
            >
              Random packet loss
            </button>
            <button
              type="button"
              className={`toggle-option ${degradeMode === 'window' ? 'is-active' : ''}`}
              onClick={() => setDegradeMode('window')}
              disabled={loading || !originalUrl}
            >
              Target specific window
            </button>
          </div>

          {degradeMode === 'percentage' ? (
            <label className="slider">
              <div className="slider-header">
                <span className="label">Simulation severity</span>
                <span className="value">{degrade}% packet loss</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={degrade}
                onChange={(e) => setDegrade(parseInt(e.target.value, 10))}
                disabled={loading}
              />
              <div className="slider-track"><span style={{ width: `${degrade}%` }} /></div>
            </label>
          ) : (
            <div className="window-controls">
              <label className="slider">
                <div className="slider-header">
                  <span className="label">Window length</span>
                  <span className="value">{windowMs} ms</span>
                </div>
                <input
                  type="range"
                  min={40}
                  max={maxWindowMs}
                  value={windowMs}
                  onChange={(e) => setWindowMs(parseInt(e.target.value, 10))}
                  step={10}
                  disabled={!audioDurationMs || loading}
                />
                <div className="slider-track"><span style={{ width: `${Math.max(0, Math.min(windowLengthTrack, 100))}%` }} /></div>
              </label>

              <label className="slider">
                <div className="slider-header">
                  <span className="label">Window offset</span>
                  <span className="value">{formatSeconds(windowStartMs)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={windowStartMax}
                  value={windowStartMs}
                  onChange={(e) => setWindowStartMs(parseInt(e.target.value, 10))}
                  disabled={!audioDurationMs || loading}
                  step={10}
                />
                <div className="slider-track"><span style={{ width: `${Math.max(0, Math.min(windowStartTrack || 0, 100))}%` }} /></div>
              </label>

              <div
                className="window-timeline"
                style={audioDurationMs ? {
                  '--window-start': `${Math.max(0, Math.min(windowStartPercent, 100))}%`,
                  '--window-end': `${Math.max(0, Math.min(windowEndPercent, 100))}%`
                } : undefined}
                aria-hidden="true"
              />

              <div className="window-meta">
                {audioDurationMs ? (
                  <>
                    <span>Clip length: {formatSeconds(audioDurationMs)}</span>
                    <span>Window spans {windowMs} ms ({formatSeconds(windowMs)})</span>
                    <span>Start at {windowStartMs} ms ({formatSeconds(windowStartMs)})</span>
                    <span>Ends at {Math.round(clampedWindowEndMs)} ms ({formatSeconds(clampedWindowEndMs)})</span>
                  </>
                ) : originalUrl ? (
                  <span>Loading clip metadata…</span>
                ) : (
                  <span>Load audio to place a targeted loss window.</span>
                )}
              </div>
            </div>
          )}

          <button className="action-button" onClick={onProcess} disabled={loading}>
            <span>{loading ? 'Processing…' : 'Process audio'}</span>
            <div className="button-glow" aria-hidden="true" />
          </button>

          {error && (
            <div className="alert" role="alert">
              <strong>Something broke:</strong> {error}
            </div>
          )}
        </section>

        <section className="analysis-section">
          <article className="stats-card">
            <h2>Nerd stats</h2>
            <ul className="metrics-grid">
              {nerdStats.map(({ label, value }) => (
                <li key={label} className="metric-item">
                  <span className="metric-label">{label}</span>
                  <span className="metric-value">{value}</span>
                </li>
              ))}
            </ul>
          </article>
          <div className="comparison-grid">
            <article className="comparison-card">
              <h2>Degraded ASR</h2>
              {renderWordColumn('degraded')}
            </article>
            <article className="comparison-card">
              <h2>Repaired text</h2>
              {renderWordColumn('repaired')}
            </article>
          </div>
        </section>

        <section className="audio-grid">
          {originalUrl && (
            <div className="audio-card">
              <h3>Original upload</h3>
              <audio controls src={originalUrl} />
              <p className="audio-caption">{file ? `Source file: ${file.name}` : 'Await audio upload.'}</p>
            </div>
          )}
          <div className={`audio-card ${loading && !degradedUrl ? 'pending-card' : ''}`}>
            <h3>Degraded audio</h3>
            {degradedUrl ? (
              <audio controls src={degradedUrl} />
            ) : (
              <div className="pending-placeholder">
                {loading ? (
                  <span className="spinner" aria-hidden="true" />
                ) : (
                  <span className="idle-icon" aria-hidden="true" />
                )}
                <p>{loading ? 'Backend is simulating packet loss…' : 'Run processing to hear the degraded version.'}</p>
              </div>
            )}
            <p className="audio-caption">
              {degradeMode === 'window' && originalUrl ? (
                <>
                  Target window: <span className="transcript-text">{windowStartMs}–{Math.round(clampedWindowEndMs)} ms</span>
                  {' '}({formatSeconds(windowStartMs)} → {formatSeconds(clampedWindowEndMs)})
                  {flaggedTokenCount > 0 &&
                    ` • ${flaggedTokenCount} tokens flagged for repair`}
                </>
              ) : (
                flaggedTokenCount
                  ? `${flaggedTokenCount} tokens flagged for repair`
                  : 'Transcript analysis will appear after processing.'
              )}
            </p>
          </div>
          <div className={`audio-card ${loading && !ttsUrl ? 'pending-card' : ''}`}>
            <h3>Synthesized repair</h3>
            {ttsUrl ? (
              <audio controls src={ttsUrl} />
            ) : (
              <div className="pending-placeholder">
                {loading ? (
                  <span className="spinner" aria-hidden="true" />
                ) : (
                  <span className="idle-icon" aria-hidden="true" />
                )}
                <p>{loading ? 'Cloning your voice and rebuilding repaired speech…' : 'Kick off processing to hear the repaired voice.'}</p>
              </div>
            )}
            <p className="audio-caption">
              {correctedTokenCount
                ? `${correctedTokenCount} corrected tokens re-synthesized`
                : 'Repaired speech will show here once processing finishes.'}
            </p>
          </div>
          {combinedUrl && (
            <div className="audio-card">
              <h3>Crossfaded blend</h3>
              <audio controls src={combinedUrl} />
              <p className="audio-caption">
                {diffCounts.matches
                  ? `Cosine blend emphasising ${diffCounts.matches} stable tokens.`
                  : 'Simple cosine crossfade between degraded and synthesized speech.'}
              </p>
            </div>
          )}
        </section>

        {elapsedMs && statusPhase === 'done' && (
          <footer className="footer-note">
            Turnaround time: {(elapsedMs / 1000).toFixed(1)}s · Whisper base · FLAN-T5 small · XTTS v2
          </footer>
        )}
      </main>
    </div>
  )
}
