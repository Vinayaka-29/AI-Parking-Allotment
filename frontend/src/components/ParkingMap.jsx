import React, { useMemo, useState } from 'react'

export default function ParkingMap({
  slots = [],
  selectedSlot,
  onSelectSlot,
  sectionFilter,
  setSectionFilter,
  onOpenCalibrator,
}) {
  const [statusFilter, setStatusFilter] = useState('ALL') // 'ALL' | 'AVAILABLE' | 'OCCUPIED' | 'RESERVED'

  // Extract unique sections dynamically from slots
  const availableSections = useMemo(() => {
    const set = new Set()
    slots.forEach((s) => {
      const sec = s.section_id || s.section
      if (sec) set.add(sec)
    })
    return ['ALL', ...Array.from(set).sort()]
  }, [slots])

  // Filter slots
  const filteredSlots = useMemo(() => {
    return slots.filter((slot) => {
      const sec = slot.section_id || slot.section
      const matchSection = sectionFilter === 'ALL' || sec === sectionFilter
      const matchStatus = statusFilter === 'ALL' || slot.status === statusFilter
      return matchSection && matchStatus
    })
  }, [slots, sectionFilter, statusFilter])

  return (
    <div className="parking-map-panel cyber-card">
      <div className="panel-header-row">
        <div className="title-group">
          <span className="panel-icon">🗺️</span>
          <div>
            <h3 className="card-title">SPATIAL PARKING GRID</h3>
            <span className="card-subtitle">
              {slots.length > 0
                ? `DYNAMIC MATRIX • ${slots.length} DETECTED BAYS`
                : 'AWAITING RECOGNITION MAP'}
            </span>
          </div>
        </div>

        {/* Filter controls */}
        <div className="map-filters-row">
          {/* Section filter */}
          {availableSections.length > 2 && (
            <div className="filter-button-group">
              {availableSections.map((sec) => (
                <button
                  key={sec}
                  type="button"
                  className={`filter-btn ${sectionFilter === sec ? 'active' : ''}`}
                  onClick={() => setSectionFilter(sec)}
                >
                  {sec === 'ALL' ? 'ALL ZONES' : `ZONE ${sec}`}
                </button>
              ))}
            </div>
          )}

          {/* Status filter */}
          <div className="filter-button-group status-pills">
            {['ALL', 'AVAILABLE', 'OCCUPIED', 'RESERVED'].map((st) => (
              <button
                key={st}
                type="button"
                className={`filter-btn status-${st.toLowerCase()} ${statusFilter === st ? 'active' : ''}`}
                onClick={() => setStatusFilter(st)}
              >
                {st}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grid container */}
      <div className="slots-grid-viewport">
        {slots.length === 0 ? (
          <div className="empty-grid-msg">
            <span className="empty-icon">🛰️</span>
            <p>NO PARKING BAYS CURRENTLY LOADED</p>
            <span>Scan an image to automatically localize parking stalls, or calibrate custom bay polygons.</span>
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
        ) : filteredSlots.length === 0 ? (
          <div className="empty-grid-msg">
            <p>NO SLOTS MATCH CURRENT FILTER CRITERIA</p>
          </div>
        ) : (
          <div className="dynamic-slots-grid">
            {filteredSlots.map((slot) => {
              const slotId = slot.slot_id || slot.id
              const isSelected = (selectedSlot?.slot_id || selectedSlot?.id) === slotId
              const isAvailable = slot.status === 'AVAILABLE'
              const isOccupied = slot.status === 'OCCUPIED'
              const isReserved = slot.status === 'RESERVED'

              const statusColorClass = isAvailable
                ? 'status-available'
                : isOccupied
                ? 'status-occupied'
                : 'status-reserved'

              const distance = slot.distance_from_entries || slot.distance || 10
              const conf = Math.round((slot.confidence || 0.95) * 100)

              return (
                <div
                  key={slotId}
                  className={`parking-slot-card ${statusColorClass} ${isSelected ? 'selected' : ''}`}
                  onClick={() => onSelectSlot && onSelectSlot(slot)}
                >
                  <div className="slot-top">
                    <span className="slot-id-badge">{slotId}</span>
                    <span className="slot-zone-pill">Z-{slot.section_id || slot.section || 'A'}</span>
                  </div>

                  <div className="slot-center-visual">
                    <span className="vehicle-symbol">
                      {isOccupied ? '🚗' : isReserved ? '🔒' : '🅿️'}
                    </span>
                    <span className="slot-status-label">{slot.status}</span>
                  </div>

                  <div className="slot-bottom">
                    <span className="slot-distance">
                      📍 {distance}m
                    </span>
                    <span className="slot-conf">
                      {conf}%
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Grid Legend Footer */}
      <div className="grid-legend-footer">
        <div className="legend-item">
          <span className="legend-dot green" />
          <span>AVAILABLE ({slots.filter(s => s.status === 'AVAILABLE').length})</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot red" />
          <span>OCCUPIED ({slots.filter(s => s.status === 'OCCUPIED').length})</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot amber" />
          <span>RESERVED ({slots.filter(s => s.status === 'RESERVED').length})</span>
        </div>
        <div className="legend-item ml-auto">
          <span>CLICK ANY BAY TO TARGET FOR DISPATCH</span>
        </div>
      </div>
    </div>
  )
}
