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
  setCameraActive,
  deviceCameraActive,
  setDeviceCameraActive,
  statusMsg,
}) {
  const fileInputRef = useRef(null)
  const deviceVideoRef = useRef(null)
  const streamRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

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
                <div className="cam-meta">CAM-01 • SECTOR-A HIGH RES OVERHEAD</div>
                <div className="cam-status">
                  <span className={`status-indicator ${cameraActive ? 'online' : 'offline'}`} />
                  {cameraActive ? 'STREAMING 1080p @ 30FPS' : 'OFFLINE'}
                </div>
              </div>

              {cameraActive ? (
                <div className="cctv-active-frame">
                  <div className="cctv-hud-grid" />
                  <div className="cctv-corner tl" />
                  <div className="cctv-corner tr" />
                  <div className="cctv-corner bl" />
                  <div className="cctv-corner br" />
                  <div className="cctv-center-target">
                    <span>LIVE AI INFERENCE ENGINE STREAMING</span>
                  </div>
                </div>
              ) : (
                <div className="cctv-standby-screen">
                  <span className="standby-icon">📡</span>
                  <h4>CCTV STREAM IN STANDBY</h4>
                  <p>Click below to initialize high-throughput stream buffer.</p>
                  <button
                    className="cyber-btn primary"
                    onClick={() => setCameraActive(true)}
                  >
                    START LIVE RTSP FEED
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

          {imagePreviewUrl && onOpenCalibrator && (
            <button
              id="calibrate-layout-btn"
              type="button"
              className="cyber-btn outline-accent"
              onClick={onOpenCalibrator}
              title="Manually draw or adjust parking stall boundaries"
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
