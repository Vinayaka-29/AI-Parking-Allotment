import React, { useState, useRef, useEffect } from 'react'
import { saveCalibratedLayout, loadCalibratedLayout } from '../utils/parkingSpaceDetector'

export default function ParkingCalibratorModal({
  imageUrl,
  existingSlots = [],
  onSaveLayout,
  onClose,
}) {
  const [slots, setSlots] = useState([])
  const [selectedSlotId, setSelectedSlotId] = useState(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPoint, setStartPoint] = useState(null)
  const [currentBox, setCurrentBox] = useState(null)
  const [gridRows, setGridRows] = useState(2)
  const [gridCols, setGridCols] = useState(8)
  const [activeSection, setActiveSection] = useState('A')

  const containerRef = useRef(null)

  // Load existing slots or previously saved calibrated layout
  useEffect(() => {
    if (existingSlots && existingSlots.length > 0) {
      setSlots(existingSlots)
    } else {
      const saved = loadCalibratedLayout()
      if (saved && saved.slots && saved.slots.length > 0) {
        setSlots(saved.slots)
      }
    }
  }, [existingSlots])

  // Get normalized coordinates (0..1) from mouse event
  const getNormalizedPoint = (e) => {
    if (!containerRef.current) return [0, 0]
    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    return [+(x).toFixed(4), +(y).toFixed(4)]
  }

  // Mouse Handlers for Drawing New Slot Box
  const handleMouseDown = (e) => {
    if (e.target.closest('.slot-del-tag')) return
    const pt = getNormalizedPoint(e)
    setIsDrawing(true)
    setStartPoint(pt)
    setCurrentBox({ x1: pt[0], y1: pt[1], x2: pt[0], y2: pt[1] })
  }

  const handleMouseMove = (e) => {
    if (!isDrawing || !startPoint) return
    const pt = getNormalizedPoint(e)
    setCurrentBox({
      x1: Math.min(startPoint[0], pt[0]),
      y1: Math.min(startPoint[1], pt[1]),
      x2: Math.max(startPoint[0], pt[0]),
      y2: Math.max(startPoint[1], pt[1]),
    })
  }

  const handleMouseUp = () => {
    if (!isDrawing || !currentBox) {
      setIsDrawing(false)
      return
    }

    const w = currentBox.x2 - currentBox.x1
    const h = currentBox.y2 - currentBox.y1

    // Ignore tiny accidental clicks (must be at least 2% of dimension)
    if (w > 0.02 && h > 0.02) {
      const nextIndex = slots.length + 1
      const slotId = `${activeSection}${String(nextIndex).padStart(2, '0')}`

      const newSlot = {
        slot_id: slotId,
        id: slotId,
        section_id: activeSection,
        section: activeSection,
        polygon: [
          [currentBox.x1, currentBox.y1],
          [currentBox.x2, currentBox.y1],
          [currentBox.x2, currentBox.y2],
          [currentBox.x1, currentBox.y2],
        ],
        distance_from_entries: nextIndex * 3,
        status: 'AVAILABLE',
        type: 'STANDARD',
      }

      setSlots(prev => [...prev, newSlot])
      setSelectedSlotId(slotId)
    }

    setIsDrawing(false)
    setStartPoint(null)
    setCurrentBox(null)
  }

  // Quick Auto-Grid Generation
  const handleGenerateGrid = () => {
    const newSlots = []
    const rows = Math.max(1, gridRows)
    const cols = Math.max(1, gridCols)

    const marginX = 0.06
    const marginY = 0.10
    const availableW = 1.0 - (2 * marginX)
    const availableH = 1.0 - (2 * marginY)

    const rowGap = 0.12
    const totalRowGaps = (rows - 1) * rowGap
    const stallH = Math.max(0.08, (availableH - totalRowGaps) / rows)

    const colGap = 0.015
    const totalColGaps = (cols - 1) * colGap
    const stallW = Math.max(0.04, (availableW - totalColGaps) / cols)

    let charCode = 65 // 'A'

    for (let r = 0; r < rows; r++) {
      const sec = String.fromCharCode(charCode)
      charCode = charCode < 90 ? charCode + 1 : 65
      const y1 = marginY + r * (stallH + rowGap)
      const y2 = y1 + stallH

      for (let c = 0; c < cols; c++) {
        const x1 = marginX + c * (stallW + colGap)
        const x2 = x1 + stallW
        const slotNum = String(c + 1).padStart(2, '0')
        const slotId = `${sec}${slotNum}`

        newSlots.push({
          slot_id: slotId,
          id: slotId,
          section_id: sec,
          section: sec,
          polygon: [
            [+x1.toFixed(4), +y1.toFixed(4)],
            [+x2.toFixed(4), +y1.toFixed(4)],
            [+x2.toFixed(4), +y2.toFixed(4)],
            [+x1.toFixed(4), +y2.toFixed(4)],
          ],
          distance_from_entries: (c + 1) * 3 + r * 10,
          status: 'AVAILABLE',
          type: 'STANDARD',
        })
      }
    }

    setSlots(newSlots)
  }

  // Delete slot
  const handleDeleteSlot = (id) => {
    setSlots(prev => prev.filter(s => (s.slot_id || s.id) !== id))
    if (selectedSlotId === id) setSelectedSlotId(null)
  }

  // Save layout
  const handleSave = () => {
    const layoutData = {
      lot_name: 'Calibrated Camera Layout',
      slots: slots.map((s, idx) => ({
        slot_id: s.slot_id || s.id || `P${idx + 1}`,
        section_id: s.section_id || s.section || 'A',
        polygon: s.polygon,
        distance_from_entries: s.distance_from_entries || (idx + 1) * 3,
        type: s.type || 'STANDARD',
      })),
      timestamp: new Date().toISOString(),
    }

    saveCalibratedLayout(layoutData)
    onSaveLayout(layoutData)
    onClose()
  }

  return (
    <div className="modal-backdrop calibrator-backdrop" onClick={onClose}>
      <div className="calibrator-modal-card" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="calibrator-header">
          <div className="calibrator-title-group">
            <span className="calibrator-icon">📐</span>
            <div>
              <h3>PARKING LAYOUT CALIBRATION STUDIO</h3>
              <p>DRAW AND CONFIGURE REAL PARKING BAYS OVER THE CAMERA SCENE</p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Toolbar */}
        <div className="calibrator-toolbar">
          <div className="toolbar-section">
            <span className="tool-lbl">QUICK STALL GRID:</span>
            <div className="tool-input-row">
              <label>ROWS:</label>
              <input
                type="number"
                min="1"
                max="8"
                value={gridRows}
                onChange={e => setGridRows(parseInt(e.target.value) || 1)}
                className="tool-number-input"
              />
              <label>BAYS/ROW:</label>
              <input
                type="number"
                min="1"
                max="24"
                value={gridCols}
                onChange={e => setGridCols(parseInt(e.target.value) || 1)}
                className="tool-number-input"
              />
              <button
                type="button"
                className="cyber-btn primary-compact"
                onClick={handleGenerateGrid}
              >
                ⚡ AUTO-FIT GRID
              </button>
            </div>
          </div>

          <div className="toolbar-section">
            <span className="tool-lbl">ACTIVE ZONE:</span>
            <div className="zone-select-row">
              {['A', 'B', 'C', 'D'].map(sec => (
                <button
                  key={sec}
                  type="button"
                  className={`sec-btn ${activeSection === sec ? 'active' : ''}`}
                  onClick={() => setActiveSection(sec)}
                >
                  ZONE {sec}
                </button>
              ))}
            </div>
          </div>

          <div className="toolbar-actions-right">
            <button
              type="button"
              className="cyber-btn ghost-compact"
              onClick={() => setSlots([])}
            >
              🗑️ CLEAR
            </button>
          </div>
        </div>

        <div className="calibrator-instruction-banner">
          <span>💡 <strong>HOW TO CALIBRATE:</strong> Use "AUTO-FIT GRID" for instant row alignment, or click and drag directly on the parking lot image to draw custom stalls. Click ✕ on any bay to remove.</span>
        </div>

        {/* Interactive Viewport Canvas */}
        <div
          ref={containerRef}
          className="calibrator-viewport"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          {imageUrl ? (
            <img
              src={imageUrl}
              alt="Parking Lot Scene for Calibration"
              className="calibrator-image"
              draggable={false}
            />
          ) : (
            <div className="calibrator-no-image">NO IMAGE LOADED</div>
          )}

          {/* SVG Overlay for existing slots and current drawing box */}
          <svg className="calibrator-svg-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none">
            {slots.map((slot) => {
              const poly = slot.polygon || []
              if (poly.length < 3) return null

              const points = poly.map(p => `${p[0] * 1000},${p[1] * 1000}`).join(' ')
              const isSelected = selectedSlotId === (slot.slot_id || slot.id)
              const id = slot.slot_id || slot.id

              return (
                <g key={id} className="calibrator-slot-g">
                  <polygon
                    points={points}
                    fill={isSelected ? 'rgba(0, 240, 255, 0.35)' : 'rgba(0, 255, 157, 0.22)'}
                    stroke={isSelected ? '#00f0ff' : '#00ff9d'}
                    strokeWidth="2.5"
                    onClick={() => setSelectedSlotId(id)}
                  />
                  {/* Label */}
                  <text
                    x={poly[0][0] * 1000 + 10}
                    y={poly[0][1] * 1000 + 24}
                    fill="#ffffff"
                    fontSize="20"
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    {id}
                  </text>
                </g>
              )
            })}

            {/* Currently dragging rectangle */}
            {isDrawing && currentBox && (
              <rect
                x={currentBox.x1 * 1000}
                y={currentBox.y1 * 1000}
                width={(currentBox.x2 - currentBox.x1) * 1000}
                height={(currentBox.y2 - currentBox.y1) * 1000}
                fill="rgba(0, 240, 255, 0.25)"
                stroke="#00f0ff"
                strokeWidth="2.5"
                strokeDasharray="6 4"
              />
            )}
          </svg>

          {/* HTML Delete buttons over each slot for easy touch/mouse deletion */}
          {slots.map((slot) => {
            const poly = slot.polygon || []
            if (poly.length < 2) return null
            const id = slot.slot_id || slot.id
            return (
              <button
                key={`del-${id}`}
                type="button"
                className="slot-del-tag"
                style={{
                  position: 'absolute',
                  left: `${poly[1][0] * 100}%`,
                  top: `${poly[0][1] * 100}%`,
                  transform: 'translate(-100%, 0)',
                }}
                onClick={(e) => {
                  e.stopPropagation()
                  handleDeleteSlot(id)
                }}
                title={`Delete ${id}`}
              >
                ✕
              </button>
            )
          })}
        </div>

        {/* Footer */}
        <div className="calibrator-footer">
          <div className="calibrator-stats-pill">
            <span>TOTAL CALIBRATED BAYS: <strong>{slots.length}</strong></span>
          </div>

          <div className="calibrator-footer-actions">
            <button
              type="button"
              className="cyber-btn ghost"
              onClick={onClose}
            >
              CANCEL
            </button>
            <button
              type="button"
              className="cyber-btn primary pulse-glow"
              onClick={handleSave}
              disabled={slots.length === 0}
            >
              💾 SAVE & APPLY AUTHORITATIVE LAYOUT ({slots.length} BAYS)
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
