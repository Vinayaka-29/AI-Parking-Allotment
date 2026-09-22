import React, { useState } from 'react'

export default function AISmartAllocation({
  slots = [],
  selectedSlot,
  onAllocate,
  isAllocating = false,
  onOpenCalibrator,
}) {
  const [licensePlate, setLicensePlate] = useState('')
  const [vehicleType, setVehicleType] = useState('sedan')
  const [allocationError, setAllocationError] = useState(null)

  const totalSlots = slots.length
  const availableSlots = slots.filter((s) => s.status === 'AVAILABLE')
  const hasAvailable = availableSlots.length > 0

  // Determine Exact Allocation Status
  let statusBadgeText = ''
  let statusBadgeClass = ''

  if (totalSlots === 0) {
    statusBadgeText = 'NO PARKING LAYOUT'
    statusBadgeClass = 'status-no-layout'
  } else if (!hasAvailable) {
    statusBadgeText = 'FACILITY FULL (100% OCCUPIED)'
    statusBadgeClass = 'status-full'
  } else {
    statusBadgeText = `${availableSlots.length} BAYS AVAILABLE`
    statusBadgeClass = 'status-available'
  }

  // Calculate best recommended slot using multi-factor scoring
  const calculateSlotScore = (slot) => {
    const dist = slot.distance_from_entries || slot.distance || 10
    const normDist = Math.max(0, 1 - (dist / 100))
    const priority = (slot.priority || 1) / 3
    const score = (normDist * 0.6) + (priority * 0.4)
    return {
      score: +score.toFixed(2),
      distance: dist,
      walkSec: Math.round(dist * 1.2),
      reasons: [
        'CONFIRMED VACANT BAY',
        `PROXIMITY: ${dist}M TO MAIN ENTRY`,
        'MINIMUM PEDESTRIAN CONGESTION',
      ],
    }
  }

  const recommendedSlot = selectedSlot && selectedSlot.status === 'AVAILABLE'
    ? selectedSlot
    : availableSlots.slice().sort((a, b) => {
        const distA = a.distance_from_entries || a.distance || 999
        const distB = b.distance_from_entries || b.distance || 999
        return distA - distB
      })[0] || null

  const recMetrics = recommendedSlot ? calculateSlotScore(recommendedSlot) : null

  const handleGeneratePlate = () => {
    const prefixes = ['KA', 'MH', 'DL', 'TS', 'TN', 'HR']
    const p = prefixes[Math.floor(Math.random() * prefixes.length)]
    const code = Math.floor(Math.random() * 89 + 10)
    const letters = 'GT'
    const num = Math.floor(Math.random() * 8999 + 1000)
    setLicensePlate(`${p}-${code}-${letters}-${num}`)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setAllocationError(null)

    if (totalSlots === 0) {
      setAllocationError('No parking layout detected. Please run analysis or calibrate bays.')
      return
    }

    if (!hasAvailable && !recommendedSlot) {
      setAllocationError('All parking spaces are currently occupied. Cannot assign ticket.')
      return
    }

    const plate = licensePlate.trim() || `KA-01-AI-${Math.floor(Math.random() * 8999 + 1000)}`
    const res = await onAllocate(plate, vehicleType, recommendedSlot)
    if (res?.error) {
      setAllocationError(res.error)
    }
  }

  return (
    <div className="ai-smart-allocation cyber-card">
      <div className="allocation-header">
        <div className="title-group">
          <span className="alloc-icon">⚡</span>
          <div>
            <h3 className="card-title">AI SMART ALLOCATION</h3>
            <span className="card-subtitle">AUTONOMOUS BAY ROUTING & PRIORITY ACCESS</span>
          </div>
        </div>
        <div className={`alloc-status-pill ${statusBadgeClass}`}>
          {statusBadgeText}
        </div>
      </div>

      {/* AI Recommendation Card */}
      {recommendedSlot ? (
        <div className="ai-recommendation-box">
          <div className="rec-header">
            <span className="rec-badge">🧠 OPTIMAL AI ASSIGNMENT</span>
            <span className="rec-slot-id">BAY {recommendedSlot.slot_id}</span>
          </div>
          <div className="rec-body">
            <div className="rec-metric">
              <span className="m-label">ZONE</span>
              <span className="m-value">ZONE {recommendedSlot.section_id || 'A'}</span>
            </div>
            <div className="rec-metric">
              <span className="m-label">DISTANCE</span>
              <span className="m-value">{recMetrics?.distance} METERS</span>
            </div>
            <div className="rec-metric">
              <span className="m-label">TRANSIT TIME</span>
              <span className="m-value">~{recMetrics?.walkSec} SEC</span>
            </div>
          </div>
          <div className="rec-reasons-list">
            {recMetrics?.reasons.map((r, i) => (
              <span key={i} className="rec-reason-item">✓ {r}</span>
            ))}
          </div>
        </div>
      ) : totalSlots === 0 ? (
        <div className="no-layout-advisory">
          <div className="advisory-title">ℹ️ PARKING LAYOUT NOT YET DEFINED</div>
          <p>Run vision analysis on the uploaded image to automatically discover stalls, or use the layout calibrator to draw parking bays.</p>
          {onOpenCalibrator && (
            <button
              type="button"
              className="cyber-btn primary-compact mt-2"
              onClick={onOpenCalibrator}
            >
              📐 OPEN LAYOUT CALIBRATOR
            </button>
          )}
        </div>
      ) : (
        <div className="full-advisory">
          <span>⚠️ All {totalSlots} parking bays are currently occupied. Incoming vehicles will be queued.</span>
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="allocation-form">
        <div className="form-group">
          <div className="label-row">
            <label htmlFor="license-plate-input">VEHICLE LICENSE PLATE</label>
            <button
              type="button"
              className="quick-generate-btn"
              onClick={handleGeneratePlate}
            >
              🎲 RANDOM PLATE
            </button>
          </div>
          <div className="input-wrapper">
            <span className="input-prefix">REG</span>
            <input
              id="license-plate-input"
              type="text"
              className="cyber-input"
              placeholder="e.g. KA-05-EV-2026"
              value={licensePlate}
              onChange={(e) => setLicensePlate(e.target.value.toUpperCase())}
            />
          </div>
        </div>

        <div className="form-group">
          <label>VEHICLE PROFILE</label>
          <div className="vehicle-type-selector">
            {[
              { id: 'sedan', label: 'SEDAN', icon: '🚗' },
              { id: 'suv', label: 'SUV', icon: '🚙' },
              { id: 'ev', label: 'EV HYBRID', icon: '⚡' },
              { id: 'luxury', label: 'LUXURY', icon: '🏎️' },
            ].map((v) => (
              <button
                key={v.id}
                type="button"
                className={`type-pill ${vehicleType === v.id ? 'active' : ''}`}
                onClick={() => setVehicleType(v.id)}
              >
                <span className="type-icon">{v.icon}</span>
                <span className="type-text">{v.label}</span>
              </button>
            ))}
          </div>
        </div>

        {allocationError && (
          <div className="allocation-error-alert">
            <span>❌ {allocationError}</span>
          </div>
        )}

        <button
          type="submit"
          className="cyber-btn primary full-width alloc-submit-btn"
          disabled={totalSlots === 0 || !hasAvailable || isAllocating}
        >
          {isAllocating ? (
            <span>GENERATING DIGITAL ACCESS PASS...</span>
          ) : totalSlots === 0 ? (
            <span>CALIBRATE OR SCAN TO ENABLE ALLOCATION</span>
          ) : !hasAvailable ? (
            <span>FACILITY FULL — NO BAYS AVAILABLE</span>
          ) : (
            <span>
              ⚡ DISPATCH SMART PASS FOR {recommendedSlot ? `BAY ${recommendedSlot.slot_id}` : 'VEHICLE'}
            </span>
          )}
        </button>
      </form>
    </div>
  )
}
