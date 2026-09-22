import React from 'react'

export default function AnalysisHistory({ history = [], onRemoveEntry, onClearAll }) {
  if (!history || history.length === 0) return null

  return (
    <div className="analysis-history-panel cyber-card">
      <div className="panel-header-row">
        <div className="title-group">
          <span className="panel-icon">⏱️</span>
          <div>
            <h3 className="card-title">RECENT ANALYSIS SESSIONS</h3>
            <span className="card-subtitle">PREVIOUS SCAN RESULTS & OCCUPANCY SNAPSHOTS</span>
          </div>
        </div>

        {onClearAll && history.length > 0 && (
          <button
            type="button"
            className="clear-history-btn"
            onClick={onClearAll}
          >
            CLEAR LOG
          </button>
        )}
      </div>

      <div className="history-cards-grid">
        {history.map((item) => {
          const total = item.overview?.total_slots ?? item.slotsCount ?? 0
          const occupied = item.overview?.occupied ?? 0
          const available = item.overview?.available ?? (total - occupied)
          const pct = total > 0 ? ((occupied / total) * 100).toFixed(1) : 0
          const time = new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

          return (
            <div key={item.id} className="history-card">
              <div className="hist-top">
                <span className="hist-time">🕒 {time}</span>
                {onRemoveEntry && (
                  <button
                    className="hist-del-btn"
                    onClick={() => onRemoveEntry(item.id)}
                    title="Remove record"
                  >
                    ✕
                  </button>
                )}
              </div>
              <div className="hist-stats-row">
                <div className="hist-stat">
                  <span className="h-val">{total}</span>
                  <span className="h-lbl">BAYS</span>
                </div>
                <div className="hist-stat">
                  <span className="h-val text-emerald">{available}</span>
                  <span className="h-lbl">FREE</span>
                </div>
                <div className="hist-stat">
                  <span className="h-val text-crimson">{occupied}</span>
                  <span className="h-lbl">BUSY</span>
                </div>
                <div className="hist-stat">
                  <span className="h-val text-amber">{pct}%</span>
                  <span className="h-lbl">OCC</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
