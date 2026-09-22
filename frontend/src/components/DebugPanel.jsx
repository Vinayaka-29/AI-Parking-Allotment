import React, { useState } from 'react'

export default function DebugPanel({ state, analysisResult }) {
  const [isOpen, setIsOpen] = useState(false)
  const [showRawJson, setShowRawJson] = useState(false)

  const imgW = analysisResult?.image?.width || '—'
  const imgH = analysisResult?.image?.height || '—'
  const vehCount = state.vehicles?.length ?? analysisResult?.vehicles?.length ?? 0
  const spaceCount = state.slots?.length ?? 0
  const occupied = state.overview?.occupied ?? 0
  const available = state.overview?.available ?? 0
  const unknown = state.overview?.unknown ?? 0
  const latency = state.telemetry?.inferenceTime ?? analysisResult?.inference_time_ms ?? '—'
  const model = 'YOLOv8 Nano (Neural Vision Tensor)'
  const layoutMode = analysisResult?.layout_mode || 'AUTOMATIC'

  return (
    <div className={`debug-panel-wrapper ${isOpen ? 'expanded' : 'collapsed'}`}>
      <button
        type="button"
        className="debug-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        title="Toggle Developer & Vision Telemetry Inspector"
      >
        <span className="debug-icon">🛠️</span>
        <span className="debug-btn-text">{isOpen ? 'CLOSE DEBUGGER' : 'DEBUG TELEMETRY'}</span>
      </button>

      {isOpen && (
        <div className="debug-panel-content cyber-card">
          <div className="debug-header">
            <h4>AI VISION & TELEMETRY INSPECTOR</h4>
            <div className="debug-tags">
              <span className="debug-tag mode">{layoutMode}</span>
              <span className="debug-tag state">{state.appState}</span>
            </div>
          </div>

          <div className="debug-grid">
            <div className="debug-item">
              <span className="dbg-lbl">FRAME DIMENSIONS</span>
              <span className="dbg-val">{imgW} × {imgH} px</span>
            </div>
            <div className="debug-item">
              <span className="dbg-lbl">DETECTED VEHICLES</span>
              <span className="dbg-val text-cyan">{vehCount}</span>
            </div>
            <div className="debug-item">
              <span className="dbg-lbl">PARKING BAYS</span>
              <span className="dbg-val">{spaceCount}</span>
            </div>
            <div className="debug-item">
              <span className="dbg-lbl">OCCUPIED / FREE</span>
              <span className="dbg-val">
                <span className="text-crimson">{occupied}</span> / <span className="text-emerald">{available}</span>
              </span>
            </div>
            <div className="debug-item">
              <span className="dbg-lbl">INFERENCE TIME</span>
              <span className="dbg-val text-amber">{latency} ms</span>
            </div>
            <div className="debug-item">
              <span className="dbg-lbl">ACTIVE MODEL</span>
              <span className="dbg-val">{model}</span>
            </div>
          </div>

          <div className="debug-actions-bar">
            <button
              type="button"
              className="cyber-btn ghost-compact"
              onClick={() => setShowRawJson(!showRawJson)}
            >
              {showRawJson ? 'HIDE RAW PAYLOAD' : 'VIEW CANONICAL JSON'}
            </button>
          </div>

          {showRawJson && (
            <pre className="debug-raw-json">
              {JSON.stringify(
                {
                  status: state.appState,
                  summary: state.overview,
                  spaces_count: spaceCount,
                  vehicles_count: vehCount,
                  slots: state.slots.map(s => ({ id: s.slot_id, status: s.status, conf: s.confidence })),
                  telemetry: state.telemetry,
                },
                null,
                2
              )}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
