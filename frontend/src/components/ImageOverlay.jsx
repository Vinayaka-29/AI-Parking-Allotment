import React, { useState } from 'react'

export default function ImageOverlay({
  imageUrl,
  annotatedImageUrl,
  slots = [],
  vehicles = [],
  onSelectSlot,
  selectedSlotId,
}) {
  const [viewMode, setViewMode] = useState('svg_hud') // 'svg_hud' | 'annotated' | 'raw'
  const [showSpaces, setShowSpaces] = useState(true)
  const [showVehicles, setShowVehicles] = useState(true)
  const [hoveredSlot, setHoveredSlot] = useState(null)

  // Determine display image
  const displayImage = (viewMode === 'annotated' && annotatedImageUrl)
    ? annotatedImageUrl
    : (imageUrl || annotatedImageUrl)

  return (
    <div className="image-overlay-wrapper">
      {/* Top Filter & Display Mode Strip */}
      <div className="overlay-controls-strip">
        <div className="overlay-mode-toggles">
          <button
            type="button"
            className={`mode-btn ${viewMode === 'svg_hud' ? 'active' : ''}`}
            onClick={() => setViewMode('svg_hud')}
          >
            HUD OVERLAY
          </button>
          {annotatedImageUrl && (
            <button
              type="button"
              className={`mode-btn ${viewMode === 'annotated' ? 'active' : ''}`}
              onClick={() => setViewMode('annotated')}
            >
              AI ANNOTATED
            </button>
          )}
          <button
            type="button"
            className={`mode-btn ${viewMode === 'raw' ? 'active' : ''}`}
            onClick={() => setViewMode('raw')}
          >
            RAW FRAME
          </button>
        </div>

        {viewMode === 'svg_hud' && (
          <div className="overlay-layer-toggles">
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={showSpaces}
                onChange={e => setShowSpaces(e.target.checked)}
              />
              <span>BAYS ({slots.length})</span>
            </label>
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={showVehicles}
                onChange={e => setShowVehicles(e.target.checked)}
              />
              <span>VEHICLES ({vehicles.length})</span>
            </label>
          </div>
        )}
      </div>

      {/* Image and Viewport Container */}
      <div className="image-viewport-container">
        {displayImage ? (
          <img
            src={displayImage}
            alt="Parking Vision Analysis Scene"
            className="base-parking-image"
          />
        ) : (
          <div className="no-image-placeholder">
            <span>NO ACTIVE FRAME STREAM</span>
          </div>
        )}

        {/* Scalable SVG HUD Layer with Normalized 0..1000 coordinate space */}
        {viewMode === 'svg_hud' && displayImage && (
          <svg
            className="svg-overlay-layer"
            viewBox="0 0 1000 1000"
            preserveAspectRatio="none"
          >
            {/* Draw Association connector lines between occupied spaces and vehicle centers */}
            {showSpaces && showVehicles && slots.map((slot) => {
              if (slot.status !== 'OCCUPIED' || !slot.polygon || slot.polygon.length < 3) return null
              // Compute center of slot
              const pts = slot.polygon.map(p => [p[0] * 1000, p[1] * 1000])
              const scx = pts.reduce((sum, p) => sum + p[0], 0) / pts.length
              const scy = pts.reduce((sum, p) => sum + p[1], 0) / pts.length

              // Find closest vehicle
              const veh = vehicles.find(v => {
                const [vx, vy, vw, vh] = v.bbox.map(n => n * 1000)
                return scx >= vx && scx <= vx + vw && scy >= vy && scy <= vy + vh
              })

              if (!veh) return null
              const [vx, vy, vw, vh] = veh.bbox.map(n => n * 1000)
              const vcx = vx + vw / 2
              const vcy = vy + vh / 2

              return (
                <line
                  key={`assoc-${slot.slot_id || slot.id}`}
                  x1={scx}
                  y1={scy}
                  x2={vcx}
                  y2={vcy}
                  stroke="#ff0055"
                  strokeWidth="1.5"
                  strokeDasharray="4 2"
                  opacity="0.8"
                />
              )
            })}

            {/* Draw Parking Space Polygons */}
            {showSpaces && slots.map((slot) => {
              const poly = slot.polygon || slot.coords || slot.coordinates
              if (!poly || poly.length < 3) return null

              const isOccupied = slot.status === 'OCCUPIED'
              const isReserved = slot.status === 'RESERVED'
              const slotId = slot.slot_id || slot.id
              const isSelected = selectedSlotId === slotId
              const isHovered = hoveredSlot === slotId

              const strokeColor = isOccupied ? '#ff0055' : isReserved ? '#ffb700' : '#00ff9d'
              const fillColor = isOccupied
                ? 'rgba(255, 0, 85, 0.22)'
                : isReserved
                ? 'rgba(255, 183, 0, 0.25)'
                : 'rgba(0, 255, 157, 0.22)'

              // Map normalized points to 0..1000 svg viewBox
              const points = poly.map(p => {
                const px = p[0] <= 1.0 ? p[0] * 1000 : p[0]
                const py = p[1] <= 1.0 ? p[1] * 1000 : p[1]
                return `${px},${py}`
              }).join(' ')

              const p0 = poly[0]
              const labelX = (p0[0] <= 1.0 ? p0[0] * 1000 : p0[0]) + 6
              const labelY = (p0[1] <= 1.0 ? p0[1] * 1000 : p0[1]) + 18
              const confPct = Math.round((slot.confidence || 0.95) * 100)

              return (
                <g
                  key={slotId}
                  className="svg-slot-group"
                  onClick={() => onSelectSlot && onSelectSlot(slot)}
                  onMouseEnter={() => setHoveredSlot(slotId)}
                  onMouseLeave={() => setHoveredSlot(null)}
                  style={{ cursor: 'pointer' }}
                >
                  <polygon
                    points={points}
                    fill={isSelected || isHovered ? 'rgba(0, 240, 255, 0.40)' : fillColor}
                    stroke={isSelected ? '#00f0ff' : strokeColor}
                    strokeWidth={isSelected || isHovered ? '3.5' : '2'}
                    strokeDasharray={isReserved ? '6 3' : 'none'}
                  />

                  {/* Slot ID & Status Badge */}
                  <rect
                    x={labelX - 4}
                    y={labelY - 14}
                    width={56}
                    height={18}
                    rx={3}
                    fill="rgba(7, 10, 18, 0.88)"
                    stroke={strokeColor}
                    strokeWidth="1"
                  />
                  <text
                    x={labelX}
                    y={labelY - 1}
                    fill="#ffffff"
                    fontSize="11"
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    {slotId}
                  </text>

                  {/* Mini status indicator */}
                  <circle
                    cx={labelX + 46}
                    cy={labelY - 5}
                    r={3}
                    fill={strokeColor}
                  />
                </g>
              )
            })}

            {/* Draw Detected Vehicle Bounding Boxes */}
            {showVehicles && vehicles.map((veh, idx) => {
              const bbox = veh.bbox || veh.pixel_bbox
              if (!bbox || bbox.length < 4) return null

              const vx = bbox[0] <= 1.0 ? bbox[0] * 1000 : bbox[0]
              const vy = bbox[1] <= 1.0 ? bbox[1] * 1000 : bbox[1]
              const vw = bbox[2] <= 1.0 ? bbox[2] * 1000 : bbox[2]
              const vh = bbox[3] <= 1.0 ? bbox[3] * 1000 : bbox[3]
              const conf = Math.round((veh.confidence || 0.94) * 100)

              return (
                <g key={`veh-${idx}`}>
                  <rect
                    x={vx}
                    y={vy}
                    width={vw}
                    height={vh}
                    fill="rgba(0, 240, 255, 0.12)"
                    stroke="#00f0ff"
                    strokeWidth="1.8"
                    strokeDasharray="4 3"
                  />
                  {/* Tag label */}
                  <rect
                    x={vx}
                    y={Math.max(0, vy - 16)}
                    width={72}
                    height={16}
                    rx={2}
                    fill="rgba(0, 240, 255, 0.88)"
                  />
                  <text
                    x={vx + 3}
                    y={Math.max(12, vy - 4)}
                    fill="#050a14"
                    fontSize="9.5"
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    VEH #{veh.id || idx + 1} {conf}%
                  </text>
                </g>
              )
            })}
          </svg>
        )}

        <div className="scanner-sweep-bar" />
      </div>
    </div>
  )
}
