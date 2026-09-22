import React from 'react'

export default function SystemStatus({
  backendConnected,
  wsConnected,
  cameraActive,
  deviceCameraActive,
  appState,
}) {
  return (
    <div className="system-status-ribbon">
      <div className="status-item">
        <span className={`status-dot ${backendConnected ? 'green' : 'amber'}`} />
        <span className="status-lbl">API BACKEND:</span>
        <span className="status-val">{backendConnected ? 'CONNECTED' : 'LOCAL ENGINE'}</span>
      </div>

      <div className="status-item">
        <span className={`status-dot ${wsConnected ? 'green' : 'neutral'}`} />
        <span className="status-lbl">WS STREAM:</span>
        <span className="status-val">{wsConnected ? 'SYNCHRONIZED' : 'POLLING'}</span>
      </div>

      <div className="status-item">
        <span className={`status-dot ${(cameraActive || deviceCameraActive) ? 'green' : 'neutral'}`} />
        <span className="status-lbl">VISION SENSOR:</span>
        <span className="status-val">
          {cameraActive ? 'RTSP CCTV' : deviceCameraActive ? 'PHONE CAM' : 'STANDBY'}
        </span>
      </div>

      <div className="status-item">
        <span className="status-dot green" />
        <span className="status-lbl">ALLOCATION CORE:</span>
        <span className="status-val">AI AUTONOMOUS</span>
      </div>
    </div>
  )
}
