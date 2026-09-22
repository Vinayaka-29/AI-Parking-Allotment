import React from 'react'

export default function ParkingSummaryBar({ overview, slots, telemetry, appState, isScanning }) {
  const isAnalyzed = appState === 'ANALYSIS_COMPLETE'
  const hasSlots = slots && slots.length > 0

  const totalSlots = isAnalyzed ? (hasSlots ? (overview?.total_slots ?? slots.length) : 0) : '—'
  const availableSlots = isAnalyzed ? (hasSlots ? (overview?.available ?? slots.filter(s => s.status === 'AVAILABLE').length) : 0) : '—'
  const occupiedSlots = isAnalyzed ? (hasSlots ? (overview?.occupied ?? slots.filter(s => s.status === 'OCCUPIED').length) : 0) : '—'
  const reservedSlots = isAnalyzed ? (hasSlots ? (overview?.reserved ?? slots.filter(s => s.status === 'RESERVED').length) : 0) : '—'

  const occupancyPct = isAnalyzed && hasSlots
    ? (overview?.occupancy_pct ?? (totalSlots > 0 ? +((occupiedSlots / totalSlots) * 100).toFixed(1) : 0))
    : 0

  return (
    <div className="parking-summary-bar">
      <div className="summary-stat-group">
        <div className="summary-stat-card total">
          <div className="stat-header">
            <span className="stat-icon">🅿️</span>
            <span className="stat-label">TOTAL BAYS</span>
          </div>
          <div className="stat-value-row">
            <span className={`stat-number ${isAnalyzed ? 'active' : 'dimmed'}`}>
              {isScanning ? <span className="scan-spin">...</span> : totalSlots}
            </span>
            <span className="stat-sub">
              {isAnalyzed ? (hasSlots ? 'GEOMETRY BOUND' : 'CALIBRATION REQ') : 'NO SCAN'}
            </span>
          </div>
        </div>

        <div className="summary-stat-card available">
          <div className="stat-header">
            <span className="stat-icon">🟢</span>
            <span className="stat-label">AVAILABLE</span>
          </div>
          <div className="stat-value-row">
            <span className={`stat-number emerald ${isAnalyzed && availableSlots > 0 ? 'active' : 'dimmed'}`}>
              {isScanning ? '...' : availableSlots}
            </span>
            <span className="stat-sub">
              {isAnalyzed ? (availableSlots > 0 ? 'READY TO ALLOCATE' : 'ZERO FREE') : 'STANDBY'}
            </span>
          </div>
        </div>

        <div className="summary-stat-card occupied">
          <div className="stat-header">
            <span className="stat-icon">🔴</span>
            <span className="stat-label">OCCUPIED</span>
          </div>
          <div className="stat-value-row">
            <span className={`stat-number crimson ${isAnalyzed && occupiedSlots > 0 ? 'active' : 'dimmed'}`}>
              {isScanning ? '...' : occupiedSlots}
            </span>
            <span className="stat-sub">DETECTED VEHICLES</span>
          </div>
        </div>

        <div className="summary-stat-card reserved">
          <div className="stat-header">
            <span className="stat-icon">🟡</span>
            <span className="stat-label">RESERVED</span>
          </div>
          <div className="stat-value-row">
            <span className={`stat-number amber ${isAnalyzed && reservedSlots > 0 ? 'active' : 'dimmed'}`}>
              {isScanning ? '...' : reservedSlots}
            </span>
            <span className="stat-sub">ACTIVE PASSES</span>
          </div>
        </div>
      </div>

      <div className="summary-occupancy-card">
        <div className="occ-header">
          <span className="occ-title">OCCUPANCY LOAD</span>
          <span className="occ-pct-badge">
            {isAnalyzed && hasSlots ? `${occupancyPct}%` : '— %'}
          </span>
        </div>
        <div className="occ-progress-track">
          <div
            className="occ-progress-fill"
            style={{
              width: isAnalyzed && hasSlots ? `${Math.min(100, Math.max(0, occupancyPct))}%` : '0%',
              backgroundColor:
                occupancyPct > 85 ? 'var(--neon-crimson)' :
                occupancyPct > 60 ? 'var(--neon-amber)' : 'var(--neon-emerald)',
              boxShadow:
                occupancyPct > 85 ? '0 0 12px rgba(255, 0, 85, 0.6)' :
                occupancyPct > 60 ? '0 0 12px rgba(255, 183, 0, 0.6)' : '0 0 12px rgba(0, 255, 157, 0.6)',
            }}
          />
        </div>
        <div className="occ-footer-meta">
          <span>
            {isAnalyzed && hasSlots
              ? `${occupiedSlots} of ${totalSlots} filled`
              : isAnalyzed
              ? 'Zero parking spaces in matrix'
              : 'Awaiting image analysis'}
          </span>
          {telemetry?.inferenceTime ? (
            <span className="inference-badge">⚡ {telemetry.inferenceTime}ms inference</span>
          ) : null}
        </div>
      </div>
    </div>
  )
}
