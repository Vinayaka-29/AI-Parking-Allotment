import React, { useState, useEffect, useRef } from 'react'
import { ParkingProvider, useParkingState, APP_STATES } from './context/ParkingContext'
import { useParkingAnalysis } from './hooks/useParkingAnalysis'
import { useWebSocket } from './hooks/useWebSocket'

import EmptyState from './components/EmptyState'
import ParkingSummaryBar from './components/ParkingSummaryBar'
import AIVisionCenter from './components/AIVisionCenter'
import ParkingMap from './components/ParkingMap'
import AISmartAllocation from './components/AISmartAllocation'
import AIConfidencePanel from './components/AIConfidencePanel'
import EventFeed from './components/EventFeed'
import AnalysisHistory from './components/AnalysisHistory'
import SystemStatus from './components/SystemStatus'
import PassModal from './components/PassModal'
import ParkingCalibratorModal from './components/ParkingCalibratorModal'
import DebugPanel from './components/DebugPanel'

import { generateParkingSnapshot } from './utils/sampleImages'

function ParkingDashboard() {
  const { state, dispatch } = useParkingState()
  const {
    uploadImage,
    analyzeImage,
    uploadAndAnalyze,
    analyzeWithLayout,
    clearAnalysis,
    allocateSlot,
    removeHistoryEntry,
  } = useParkingAnalysis()

  // Initialize WebSocket connection
  useWebSocket()

  const [selectedSlot, setSelectedSlot] = useState(null)
  const [isAllocating, setIsAllocating] = useState(false)
  const [isCalibratorOpen, setIsCalibratorOpen] = useState(false)
  const hiddenFileInputRef = useRef(null)

  // Current system clock
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString())
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString())
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  // Handle file select
  const handleFileSelect = (file) => {
    if (!file) return
    uploadImage(file)
  }

  // Handle run analysis
  const handleRunAnalysis = () => {
    if (state.uploadedFile || state.imagePreviewUrl) {
      analyzeImage(state.uploadedFile)
    }
  }

  // Handle calibrated layout save
  const handleSaveCalibratedLayout = (layoutData) => {
    analyzeWithLayout(layoutData)
  }

  // Handle benchmark demo loading (explicit DEMO MODE)
  const handleSampleDemo = async (scenario = 'half') => {
    try {
      const blob = await generateParkingSnapshot(scenario)
      const demoFile = new File([blob], `benchmark_${scenario}_lot.jpg`, { type: 'image/jpeg' })
      await uploadAndAnalyze(demoFile)
    } catch (err) {
      console.error('Demo generation failed:', err)
    }
  }

  // Handle allocation dispatch
  const handleAllocate = async (plate, vehicleType, targetSlot) => {
    setIsAllocating(true)
    try {
      const res = await allocateSlot(plate, vehicleType, targetSlot)
      return res
    } finally {
      setIsAllocating(false)
    }
  }

  const isScanning = state.appState === APP_STATES.ANALYZING
  const hasImage = Boolean(state.uploadedFile || state.imagePreviewUrl || state.annotatedImage)
  const isStandby = state.appState === APP_STATES.NO_ANALYSIS && !hasImage && state.activeTab === 'upload'

  return (
    <div className="cyber-dashboard-app">
      {/* Background Cyber Grid Effects */}
      <div className="cyber-grid-ambient" />
      <div className="cyber-glow-orb orb-1" />
      <div className="cyber-glow-orb orb-2" />

      {/* Hidden file input for global triggers */}
      <input
        type="file"
        ref={hiddenFileInputRef}
        style={{ display: 'none' }}
        accept="image/*"
        onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
      />

      {/* Top Cyber Navigation Bar */}
      <header className="cyber-nav-header">
        <div className="nav-brand-group">
          <div className="brand-logo-hex">
            <span className="logo-glitch">🅿️</span>
          </div>
          <div className="brand-text">
            <div className="brand-title">
              <span className="brand-accent">PARK</span>-AI // NEURAL INTELLIGENCE
            </div>
            <div className="brand-subtitle">
              AUTONOMOUS SPATIAL VISION & DYNAMIC ALLOCATION MATRIX
            </div>
          </div>
        </div>

        <div className="nav-controls-group">
          <div className="hud-clock-pill">
            <span className="clock-icon">🕒</span>
            <span className="clock-digits">{currentTime}</span>
          </div>

          <div className={`nav-state-badge state-${state.appState.toLowerCase()}`}>
            <span className="state-dot" />
            <span className="state-label">
              {state.appState === APP_STATES.NO_ANALYSIS && 'STANDBY • NO SCAN'}
              {state.appState === APP_STATES.IMAGE_UPLOADED && 'IMAGE LOADED • READY'}
              {state.appState === APP_STATES.ANALYZING && 'ANALYZING TENSORS'}
              {state.appState === APP_STATES.ANALYSIS_COMPLETE && (
                state.slots.length > 0
                  ? `AI MATRIX ACTIVE (${state.slots.length} BAYS)`
                  : 'NO PARKING LAYOUT'
              )}
              {state.appState === APP_STATES.ANALYSIS_ERROR && 'DETECTION ADVISORY'}
              {state.appState === APP_STATES.LIVE_CAMERA && 'CCTV REAL-TIME'}
            </span>
          </div>

          <div className="nav-action-buttons">
            <button
              id="nav-upload-trigger"
              className="cyber-btn primary-compact"
              onClick={() => hiddenFileInputRef.current?.click()}
            >
              📁 UPLOAD IMAGE
            </button>

            {hasImage && (
              <button
                id="nav-calibrate-trigger"
                className="cyber-btn outline-compact"
                onClick={() => setIsCalibratorOpen(true)}
                title="Open visual parking layout calibrator"
              >
                📐 CALIBRATE BAYS
              </button>
            )}

            <button
              id="nav-demo-trigger"
              className="cyber-btn demo-compact"
              onClick={() => handleSampleDemo('half')}
              title="Loads calibrated benchmark parking image for immediate evaluation"
            >
              <span className="demo-tag">DEMO</span> BENCHMARK
            </button>

            {hasImage && (
              <button
                id="nav-reset-trigger"
                className="cyber-btn ghost-compact"
                onClick={clearAnalysis}
                title="Clear current analysis"
              >
                🔄 RESET
              </button>
            )}
          </div>
        </div>
      </header>

      {/* System Telemetry Ribbon */}
      <SystemStatus
        backendConnected={state.backendConnected}
        wsConnected={state.wsConnected}
        cameraActive={state.cameraActive}
        deviceCameraActive={state.deviceCameraActive}
        appState={state.appState}
      />

      {/* Main Container */}
      <main className="dashboard-main-content">
        {/* Statistics Summary Bar (Dynamic: dashes when unanalyzed) */}
        <ParkingSummaryBar
          overview={state.overview}
          slots={state.slots}
          telemetry={state.telemetry}
          appState={state.appState}
          isScanning={isScanning}
        />

        {/* Empty Standby Hero when zero image loaded */}
        {isStandby ? (
          <EmptyState
            onUploadClick={() => hiddenFileInputRef.current?.click()}
            onDemoClick={() => handleSampleDemo('half')}
            onCameraClick={() => {
              dispatch({ type: 'SET_ACTIVE_TAB', payload: 'stream' })
              dispatch({ type: 'SET_CAMERA_ACTIVE', payload: true })
            }}
          />
        ) : (
          /* Active Analysis & Workspace Grid */
          <div className="active-intelligence-grid">
            {/* Left Column: Vision Center & Confidence */}
            <div className="grid-column-left">
              <AIVisionCenter
                appState={state.appState}
                uploadedFile={state.uploadedFile}
                imagePreviewUrl={state.imagePreviewUrl}
                annotatedImage={state.annotatedImage}
                slots={state.slots}
                vehicles={state.vehicles}
                analysisProgress={state.analysisProgress}
                analysisError={state.analysisError}
                isScanning={isScanning}
                onFileUpload={handleFileSelect}
                onRunAnalysis={handleRunAnalysis}
                onClearAnalysis={clearAnalysis}
                onOpenCalibrator={() => setIsCalibratorOpen(true)}
                onSampleDemoClick={() => handleSampleDemo('crowded')}
                activeTab={state.activeTab}
                setActiveTab={(tab) => dispatch({ type: 'SET_ACTIVE_TAB', payload: tab })}
                cameraActive={state.cameraActive}
                setCameraActive={(active) => dispatch({ type: 'SET_CAMERA_ACTIVE', payload: active })}
                deviceCameraActive={state.deviceCameraActive}
                setDeviceCameraActive={(active) => dispatch({ type: 'SET_DEVICE_CAMERA_ACTIVE', payload: active })}
                statusMsg={state.statusMsg}
              />

              <AIConfidencePanel
                confidence={state.confidence}
                telemetry={state.telemetry}
                slots={state.slots}
                vehicles={state.vehicles}
              />
            </div>

            {/* Right Column: Spatial Map & Allocation Engine */}
            <div className="grid-column-right">
              <ParkingMap
                slots={state.slots}
                selectedSlot={selectedSlot}
                onSelectSlot={setSelectedSlot}
                sectionFilter={state.sectionFilter}
                setSectionFilter={(sec) => dispatch({ type: 'SET_SECTION_FILTER', payload: sec })}
                onOpenCalibrator={() => setIsCalibratorOpen(true)}
              />

              <AISmartAllocation
                slots={state.slots}
                selectedSlot={selectedSlot}
                onAllocate={handleAllocate}
                isAllocating={isAllocating}
                onOpenCalibrator={() => setIsCalibratorOpen(true)}
              />

              <EventFeed events={state.events} />
            </div>
          </div>
        )}

        {/* Previous Analysis Sessions History */}
        {state.analysisHistory.length > 0 && (
          <AnalysisHistory
            history={state.analysisHistory}
            onRemoveEntry={removeHistoryEntry}
          />
        )}
      </main>

      {/* Visual Calibrator Modal */}
      {isCalibratorOpen && (
        <ParkingCalibratorModal
          imageUrl={state.imagePreviewUrl || state.annotatedImage}
          existingSlots={state.slots}
          onSaveLayout={handleSaveCalibratedLayout}
          onClose={() => setIsCalibratorOpen(false)}
        />
      )}

      {/* Digital Access Pass Modal */}
      {state.issuedTicket && (
        <PassModal
          ticket={state.issuedTicket}
          onClose={() => dispatch({ type: 'SET_ISSUED_TICKET', payload: null })}
        />
      )}

      {/* Developer & Telemetry Inspector Panel */}
      <DebugPanel
        state={state}
        analysisResult={state.analysisResult}
      />

      {/* Cyberpunk Footer */}
      <footer className="cyber-footer">
        <div className="footer-left">
          <span className="dot pulse green" />
          <span>AUTONOMOUS INFERENCE PROTOCOL v2.5</span>
        </div>
        <div className="footer-center">
          <span>ZERO HARDCODED PARKING SLOTS • DYNAMIC COMPUTER VISION MATRIX</span>
        </div>
        <div className="footer-right">
          <span>AI PARKING INTELLIGENCE MATRIX</span>
        </div>
      </footer>
    </div>
  )
}

export default function App() {
  return (
    <ParkingProvider>
      <ParkingDashboard />
    </ParkingProvider>
  )
}
