import React, { useRef, useState, useEffect } from 'react'
import ImageOverlay from './ImageOverlay'

export default function AIVisionCenter({
  appState,
  uploadedFile,
  imagePreviewUrl,
  annotatedImage,
  slots,
  vehicles,
  analysisProgress,
  analysisError,
  isScanning,
  onFileUpload,
  onRunAnalysis,
  onClearAnalysis,
  onOpenCalibrator,
  onSampleDemoClick,
  activeTab,
  setActiveTab,
  cameraActive,
  cameraSnapshotUrl,
  cameraLayoutSource,
  onLiveFrameAnalysis,
  setCameraActive,
  deviceCameraActive,
  setDeviceCameraActive,
  statusMsg,
}) {
  const fileInputRef = useRef(null)
  const deviceVideoRef = useRef(null)
  const streamRef = useRef(null)
  const cameraFrameBusyRef = useRef(false)
  const lastCameraFrameAnalysisRef = useRef(0)
  const [dragOver, setDragOver] = useState(false)
  const [cameraSnapshotOnline, setCameraSnapshotOnline] = useState(false)
  const [cameraPixelsReadable, setCameraPixelsReadable] = useState(null)
  const [snapshotRefresh, setSnapshotRefresh] = useState(Date.now())
  const [cameraOverlayMode, setCameraOverlayMode] = useState('both')
  const [cameraImageDimensions, setCameraImageDimensions] = useState(null)

  // Some cameras accept browser image requests even when the backend cannot
  // reach them. Poll the JPEG directly as a display fallback.
  useEffect(() => {
    if (!cameraActive) {
      setCameraSnapshotOnline(false)
      return
    }

    setCameraPixelsReadable(null)
    const refreshTimer = window.setInterval(() => setSnapshotRefresh(Date.now()), 2000)
    return () => {
      window.clearInterval(refreshTimer)
    }
  }, [cameraActive])

  const analyzeDirectCameraFrame = (event) => {
    const image = event.currentTarget
    if (cameraFrameBusyRef.current || !onLiveFrameAnalysis || !image.naturalWidth || !image.naturalHeight) return
    if (Date.now() - lastCameraFrameAnalysisRef.current < 3000) return

    try {
      const canvas = document.createElement('canvas')
      canvas.width = image.naturalWidth
      canvas.height = image.naturalHeight
      canvas.getContext('2d').drawImage(image, 0, 0)
      cameraFrameBusyRef.current = true
      lastCameraFrameAnalysisRef.current = Date.now()
      canvas.toBlob(async (blob) => {
        try {
          if (!blob) throw new Error('Could not encode live frame')
          await onLiveFrameAnalysis(new File([blob], `live-camera-${Date.now()}.jpg`, { type: 'image/jpeg' }))
          setCameraPixelsReadable(true)
        } catch (error) {
          setCameraPixelsReadable(false)
          console.warn('Live CCTV frame analysis unavailable:', error)
        } finally {
          cameraFrameBusyRef.current = false
        }
      }, 'image/jpeg', 0.88)
    } catch (error) {
      cameraFrameBusyRef.current = false
      setCameraPixelsReadable(false)
      console.warn('Camera image blocks browser pixel access:', error)
    }
  }

  const directSnapshotSrc = cameraSnapshotUrl
    ? `${cameraSnapshotUrl}${cameraSnapshotUrl.includes('?') ? '&' : '?'}_=${snapshotRefresh}`
    : ''
  const showBays = cameraOverlayMode === 'bays' || cameraOverlayMode === 'both'
  const showVehicles = cameraOverlayMode === 'vehicles' || cameraOverlayMode === 'both'
  const visibleSource = cameraSnapshotOnline ? 'DIRECT SNAPSHOT + FRONTEND HUD' : 'BACKEND MJPEG (COMBINED)'

  // Manage Device Camera stream
  useEffect(() => {
    if (deviceCameraActive) {
      navigator.mediaDevices?.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      })
      .then((stream) => {
        streamRef.current = stream
        if (deviceVideoRef.current) {
          deviceVideoRef.current.srcObject = stream
          deviceVideoRef.current.play()
        }
      })
      .catch((err) => {
        console.error('Camera access error:', err)
        setDeviceCameraActive(false)
      })
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
        streamRef.current = null
      }
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
    }
  }, [deviceCameraActive, setDeviceCameraActive])

  // Capture frame from active camera
  const captureCameraFrame = () => {
    const video = deviceVideoRef.current
    if (!video) return

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `cam_capture_${Date.now()}.jpg`, { type: 'image/jpeg' })
        onFileUpload(file)
        setActiveTab('upload')
        setDeviceCameraActive(false)
      }
    }, 'image/jpeg', 0.92)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0])
    }
  }

  const occupiedCount = slots.filter(s => s.status === 'OCCUPIED').length
  const availableCount = slots.filter(s => s.status === 'AVAILABLE').length

  return (
    <div className="ai-vision-center cyber-card">
      <div className="vision-header">
        <div className="vision-title-group">
          <span className="vision-icon">👁️</span>
          <div>
            <h3 className="card-title">YOLO VISION INTELLIGENCE CENTER</h3>
            <span className="card-subtitle">REAL-TIME VEHICLE & SPACE RECOGNITION PIPELINE</span>
          </div>
        </div>

        {/* Source Navigation Tabs */}
        <div className="vision-source-tabs">
          <button
            type="button"
            className={`source-tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📷 CAPTURE / UPLOAD
          </button>
          <button
            type="button"
            className={`source-tab-btn ${activeTab === 'stream' ? 'active' : ''}`}
            onClick={() => setActiveTab('stream')}
          >
            📡 LIVE RTSP / CCTV
          </button>
          <button
            type="button"
            className={`source-tab-btn ${activeTab === 'device' ? 'active' : ''}`}
            onClick={() => setActiveTab('device')}
          >
            📱 PHONE CAM
          </button>
        </div>
      </div>

      {/* Main Viewport Content */}
      <div className="vision-viewport-box">
        {/* TAB 1: UPLOAD & IMAGE ANALYSIS */}
        {activeTab === 'upload' && (
          <div className="upload-tab-pane">
            {imagePreviewUrl || annotatedImage ? (
              <div className="preview-active-view">
                <ImageOverlay
                  imageUrl={imagePreviewUrl}
                  annotatedImageUrl={annotatedImage}
                  slots={slots}
                  vehicles={vehicles}
                />

                {/* Status Overlay Badge */}
                <div className="viewport-hud-tag">
                  <span className="dot pulse" />
                  {isScanning
                    ? 'ANALYZING TENSORS...'
                    : slots.length > 0
                    ? `${slots.length} BAYS • ${occupiedCount} OCCUPIED • ${availableCount} FREE`
                    : vehicles.length > 0
                    ? `${vehicles.length} VEHICLES DETECTED`
                    : 'FRAME LOADED • CLICK RUN ANALYSIS'}
                </div>
              </div>
            ) : (
              <div
                className={`dropzone-box ${dragOver ? 'drag-over' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={(e) => e.target.files?.[0] && onFileUpload(e.target.files[0])}
                />
                <div className="dropzone-icon">📥</div>
                <div className="dropzone-text-primary">DROP PARKING IMAGE HERE OR CLICK TO BROWSE</div>
                <div className="dropzone-text-secondary">SUPPORTS HIGH-RESOLUTION JPG, PNG, WEBP (AERIAL / CCTV)</div>
                <div className="dropzone-tags">
                  <span>YOLOv8 DETECTOR</span>
                  <span>GEOMETRIC STALL RECONSTRUCTION</span>
                  <span>IoU OVERLAP MAPPING</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: CCTV / RTSP FEED */}
        {activeTab === 'stream' && (
          <div className="stream-tab-pane">
            <div className="cctv-viewport">
              <div className="cctv-header-bar">
                <div className="cam-meta">
                  📍 CAM-ASAHI-JP &nbsp;•&nbsp; Asahi Parking Lot, Chiba, Japan
                  &nbsp;•&nbsp; Hi3516 · {cameraImageDimensions ? `${cameraImageDimensions.width}×${cameraImageDimensions.height}` : 'DIMENSIONS PENDING'}
                </div>
                <div className="cam-status">
                  <span className={`status-indicator ${cameraActive ? 'online' : 'offline'}`} />
                  {cameraActive ? (cameraSnapshotOnline ? 'LIVE CAMERA SNAPSHOT' : 'CONNECTING TO CAMERA') : 'OFFLINE'}
                </div>
              </div>

              {cameraActive ? (
                <div className="cctv-active-frame">
                  <div className="cctv-debug-toolbar">
                    <span>LAYOUT: {cameraLayoutSource}</span>
                    <span>VISIBLE: {visibleSource}</span>
                    <div className="cctv-overlay-toggle" role="group" aria-label="Live overlay mode">
                      {[
                        ['bays', 'BAYS ONLY'],
                        ['vehicles', 'VEHICLES ONLY'],
                        ['both', 'BOTH'],
                      ].map(([mode, label]) => (
                        <button
                          key={mode}
                          type="button"
                          className={cameraOverlayMode === mode ? 'active' : ''}
                          onClick={() => setCameraOverlayMode(mode)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                  {/* Live MJPEG stream from backend — YOLO + slot overlays rendered server-side */}
                  {directSnapshotSrc && (
                    <img
                      className="camera-source-probe"
                      src={directSnapshotSrc}
                      alt=""
                      onLoad={() => setCameraSnapshotOnline(true)}
                      onError={() => setCameraSnapshotOnline(false)}
                    />
                  )}
                  {directSnapshotSrc && (
                    <img
                      className="camera-pixel-probe"
                      crossOrigin="anonymous"
                      src={directSnapshotSrc}
                      alt=""
                      onLoad={analyzeDirectCameraFrame}
                      onError={() => setCameraPixelsReadable(false)}
                    />
                  )}
                  {cameraSnapshotOnline ? (
                    <>
                      <img
                        src={directSnapshotSrc}
                        alt="Direct live Asahi parking camera snapshot"
                        className="cctv-stream cctv-direct-snapshot"
                        onLoad={(event) => setCameraImageDimensions({
                          width: event.currentTarget.naturalWidth,
                          height: event.currentTarget.naturalHeight,
                        })}
                        onError={() => setCameraSnapshotOnline(false)}
                      />
                      <img src="/api/camera/feed" alt="" className="camera-server-feed-probe" />
                    </>
                  ) : (
                    <img
                      src="/api/camera/feed"
                      alt="Asahi parking camera connection status"
                      className="cctv-stream"
                      onLoad={(event) => setCameraImageDimensions({
                        width: event.currentTarget.naturalWidth,
                        height: event.currentTarget.naturalHeight,
                      })}
                    />
                  )}
                  {cameraSnapshotOnline && <svg className="live-camera-analysis-overlay" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid meet">
                    {showBays && slots.map((slot) => {
                      const polygon = slot.polygon || []
                      if (polygon.length < 3) return null
                      const points = polygon.map(([x, y]) => `${x <= 1 ? x * 1920 : x},${y <= 1 ? y * 1080 : y}`).join(' ')
                      const color = slot.status === 'OCCUPIED' ? '#ff3159' : slot.status === 'RESERVED' ? '#ffbd32' : slot.status === 'UNKNOWN' ? '#9aa8ba' : '#20e88a'
                      const [rawX, rawY] = polygon[0]
                      const lx = rawX <= 1 ? rawX * 1920 : rawX
                      const ly = rawY <= 1 ? rawY * 1080 : rawY
                      return <g key={slot.slot_id}><polygon points={points} fill={`${color}30`} stroke={color} strokeWidth="3" /><text x={lx} y={ly - 6} fill="white" fontSize="24" stroke="#08111b" strokeWidth="5" paintOrder="stroke">{slot.slot_id}</text></g>
                    })}
                    {showVehicles && vehicles.map((vehicle, index) => {
                      const [x1, y1, x2, y2] = vehicle.bbox || []
                      if (![x1, y1, x2, y2].every(Number.isFinite)) return null
                      return <g key={`camera-vehicle-${vehicle.tracking_id || index}`}><rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="rgba(0,240,255,.12)" stroke="#00f0ff" strokeWidth="4" /><text x={x1} y={Math.max(24, y1 - 8)} fill="#00f0ff" fontSize="24">{vehicle.class_name} {Math.round((vehicle.confidence || 0) * 100)}%</text></g>
                    })}
                  </svg>}
                  <div className="cctv-hud-grid" />
                  <div className="cctv-corner tl" />
                  <div className="cctv-corner tr" />
                  <div className="cctv-corner bl" />
                  <div className="cctv-corner br" />
                  <div className="cctv-center-target">
                    <span>{cameraPixelsReadable === false ? 'CAMERA VISIBLE · BROWSER PIXEL ACCESS BLOCKED' : `${vehicles.length} YOLO VEHICLES · ASAHI JP CCTV`}</span>
                  </div>
                  {/* Slot badge */}
                  <div className="viewport-hud-tag" style={{ bottom: '10px', top: 'auto' }}>
                    <span className="dot pulse" />
                    {slots.length > 0
                      ? `${slots.length} CALIBRATED BAYS • ${occupiedCount} OCCUPIED • ${availableCount} FREE`
                      : 'DETECTING PARKING SLOTS…'}
                  </div>
                </div>
              ) : (
                <div className="cctv-standby-screen">
                  <span className="standby-icon">📡</span>
                  <h4>ASAHI PARKING CCTV — STANDBY</h4>
                  <p style={{ fontSize: '0.78rem', opacity: 0.7, margin: '4px 0 12px' }}>
                    Real live camera · Asahi, Chiba, Japan (insecam #945438)<br />
                    Hi3516 · 1920×1080 · Pull-mode JPEG @ 3 fps<br />
                    YOLOv8-Nano with calibrated slot polygons
                  </p>
                  <button
                    className="cyber-btn primary"
                    onClick={() => setCameraActive(true)}
                  >
                    START LIVE ASAHI CCTV FEED
                  </button>
                </div>
              )}
            </div>
          </div>
        )}


        {/* TAB 3: PHONE / WEBCAM */}
        {activeTab === 'device' && (
          <div className="device-tab-pane">
            <div className="device-camera-viewport">
              {deviceCameraActive ? (
                <div className="video-container">
                  <video
                    ref={deviceVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className="device-video-element"
                  />
                  <div className="cam-controls-overlay">
                    <button
                      className="cyber-btn primary pulse-glow"
                      onClick={captureCameraFrame}
                    >
                      📸 CAPTURE FRAME & ANALYZE
                    </button>
                    <button
                      className="cyber-btn outline"
                      onClick={() => setDeviceCameraActive(false)}
                    >
                      STOP CAMERA
                    </button>
                  </div>
                </div>
              ) : (
                <div className="device-cam-standby">
                  <span className="cam-icon">📱</span>
                  <h4>DEVICE CAMERA SENSOR</h4>
                  <p>Use your mobile or laptop webcam to scan a parking scenario.</p>
                  <button
                    className="cyber-btn primary"
                    onClick={() => setDeviceCameraActive(true)}
                  >
                    ACTIVATE REAR / DEVICE CAMERA
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Analysis Progress Stepper (Visible while scanning) */}
      {isScanning && (
        <div className="analysis-progress-stepper">
          <div className="stepper-title">
            <span className="pulse-dot" /> NEURAL VISION PIPELINE IN EXECUTION
          </div>
          <div className="steps-row">
            {analysisProgress && analysisProgress.length > 0 ? (
              analysisProgress.map((step, idx) => (
                <div
                  key={idx}
                  className={`step-item ${step.status === 'complete' ? 'done' : step.status === 'active' ? 'current' : ''}`}
                >
                  <div className="step-circle">
                    {step.status === 'complete' ? '✓' : idx + 1}
                  </div>
                  <span className="step-label">{step.label}</span>
                </div>
              ))
            ) : (
              <span>Running vehicle & spatial localization...</span>
            )}
          </div>
        </div>
      )}

      {/* Error notification if analysis failed */}
      {analysisError && (
        <div className="vision-error-banner">
          <span className="err-icon">⚠️</span>
          <div className="err-body">
            <strong>VISION RECOGNITION ADVISORY:</strong>
            <p>{analysisError}</p>
          </div>
        </div>
      )}

      {/* Action Footer Bar */}
      <div className="vision-footer-bar">
        <div className="status-readout">
          <span className="live-dot" />
          <span className="status-text">{statusMsg}</span>
        </div>

        <div className="vision-action-buttons">
          {imagePreviewUrl && !isScanning && (
            <button
              id="run-analysis-btn"
              className="cyber-btn primary pulse-glow"
              onClick={onRunAnalysis}
            >
              🚀 RUN YOLO VISION ANALYSIS
            </button>
          )}

          {(imagePreviewUrl || (cameraActive && cameraSnapshotUrl)) && onOpenCalibrator && (
            <button
              id="calibrate-layout-btn"
              type="button"
              className="cyber-btn outline-accent"
              onClick={onOpenCalibrator}
              title="Manually draw or adjust parking stall boundaries on the current scene"
            >
              📐 CALIBRATE LAYOUT
            </button>
          )}

          {imagePreviewUrl && (
            <button
              className="cyber-btn ghost"
              onClick={onClearAnalysis}
              title="Reset frame"
            >
              🔄 CLEAR
            </button>
          )}

          <button
            className="cyber-btn ghost"
            onClick={() => fileInputRef.current?.click()}
          >
            📁 CHANGE IMAGE
          </button>

          {onSampleDemoClick && (
            <button
              className="cyber-btn demo-btn"
              onClick={onSampleDemoClick}
              title="Loads high-contrast aerial parking benchmark"
            >
              <span className="demo-tag">DEMO</span> SYNTHETIC PRESET
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
