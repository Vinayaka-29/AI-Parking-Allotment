import React from 'react'

export default function EmptyState({ onUploadClick, onDemoClick, onCameraClick }) {
  return (
    <div className="empty-state-container">
      <div className="empty-state-radar">
        <div className="radar-sweep" />
        <div className="radar-grid" />
        <div className="radar-crosshair-h" />
        <div className="radar-crosshair-v" />
        <div className="radar-ring r1" />
        <div className="radar-ring r2" />
        <div className="radar-ring r3" />
        <div className="radar-center-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="3" y="3" width="18" height="18" rx="3" stroke="#00f0ff" strokeDasharray="3 2" />
            <path d="M9 17V7h4a3 3 0 0 1 0 6H9" stroke="#00f0ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="17" cy="17" r="2" fill="#00ff9d" />
          </svg>
        </div>
      </div>

      <div className="empty-state-content">
        <div className="hud-badge-ghost">STANDBY • AWAITING INPUT</div>
        <h2 className="empty-state-title">
          NO PARKING ANALYSIS LOADED
        </h2>
        <p className="empty-state-desc">
          Total parking bays and occupancy are determined dynamically through AI computer vision.
          Upload a parking lot capture, connect a live video feed, or load a benchmark demo to start real-time slot allocation.
        </p>

        <div className="empty-state-actions">
          <button
            id="empty-upload-btn"
            className="cyber-btn primary pulse-glow"
            onClick={onUploadClick}
          >
            <span className="btn-icon">📁</span>
            <span>UPLOAD PARKING CAPTURE</span>
          </button>

          <button
            id="empty-camera-btn"
            className="cyber-btn outline"
            onClick={onCameraClick}
          >
            <span className="btn-icon">📹</span>
            <span>CONNECT LIVE CCTV / CAM</span>
          </button>

          {onDemoClick && (
            <button
              id="empty-demo-btn"
              className="cyber-btn ghost-subtle"
              onClick={onDemoClick}
            >
              <span className="demo-pill-badge">DEMO</span>
              <span>LOAD BENCHMARK PRESET</span>
            </button>
          )}
        </div>

        <div className="empty-state-features">
          <div className="feature-pill">
            <span className="dot cyan" /> Zero Hardcoded Slots
          </div>
          <div className="feature-pill">
            <span className="dot emerald" /> YOLOv8 Spatial Detection
          </div>
          <div className="feature-pill">
            <span className="dot amber" /> Dynamic IoU Space Mapping
          </div>
        </div>
      </div>
    </div>
  )
}
