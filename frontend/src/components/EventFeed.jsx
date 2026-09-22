import React, { useState } from 'react'

export default function EventFeed({ events = [] }) {
  const [filter, setFilter] = useState('ALL') // 'ALL' | 'ALLOC' | 'STATUS'

  const filteredEvents = events.filter((ev) => {
    if (filter === 'ALLOC') return ev.event_type?.includes('ASSIGN') || ev.event_type?.includes('TICKET') || ev.type?.includes('ASSIGN')
    if (filter === 'STATUS') return ev.event_type?.includes('STATUS') || ev.type?.includes('STATUS')
    return true
  })

  return (
    <div className="event-feed-panel cyber-card">
      <div className="panel-header-row">
        <div className="title-group">
          <span className="panel-icon">📜</span>
          <div>
            <h3 className="card-title">LIVE TELEMETRY EVENT STREAM</h3>
            <span className="card-subtitle">SYSTEM LOGS & OCCUPANCY TRANSITIONS</span>
          </div>
        </div>

        <div className="filter-button-group">
          <button
            type="button"
            className={`filter-btn ${filter === 'ALL' ? 'active' : ''}`}
            onClick={() => setFilter('ALL')}
          >
            ALL ({events.length})
          </button>
          <button
            type="button"
            className={`filter-btn ${filter === 'ALLOC' ? 'active' : ''}`}
            onClick={() => setFilter('ALLOC')}
          >
            DISPATCHES
          </button>
          <button
            type="button"
            className={`filter-btn ${filter === 'STATUS' ? 'active' : ''}`}
            onClick={() => setFilter('STATUS')}
          >
            SENSORS
          </button>
        </div>
      </div>

      <div className="events-stream-box">
        {filteredEvents.length === 0 ? (
          <div className="empty-events-msg">
            <span>📡 NO TELEMETRY EVENTS RECORDED YET</span>
          </div>
        ) : (
          <div className="events-list">
            {filteredEvents.slice(0, 30).map((ev, idx) => {
              const type = ev.event_type || ev.type || 'INFO'
              const isAlloc = type.includes('ASSIGN') || type.includes('TICKET')
              const isOccupied = ev.status === 'OCCUPIED' || type.includes('ENTER')
              const isAvailable = ev.status === 'AVAILABLE' || type.includes('LEAVE') || type.includes('LEFT')

              const icon = isAlloc ? '🎫' : isOccupied ? '🚗' : isAvailable ? '🟢' : '⚡'
              const colorClass = isAlloc ? 'text-amber' : isOccupied ? 'text-crimson' : isAvailable ? 'text-emerald' : 'text-cyan'

              const timeStr = ev.timestamp
                ? new Date(ev.timestamp).toLocaleTimeString()
                : 'JUST NOW'

              return (
                <div key={ev.event_id || idx} className="event-entry-row">
                  <span className="ev-icon">{icon}</span>
                  <div className="ev-content">
                    <div className="ev-header">
                      <span className={`ev-type ${colorClass}`}>{type.replace(/_/g, ' ')}</span>
                      <span className="ev-time">{timeStr}</span>
                    </div>
                    <div className="ev-details">
                      {ev.slot_id && <span className="ev-slot">BAY {ev.slot_id}</span>}
                      {ev.vehicle_id && <span className="ev-plate">[{ev.vehicle_id}]</span>}
                      {ev.status && <span className="ev-status">→ {ev.status}</span>}
                      {ev.confidence && (
                        <span className="ev-conf">
                          ({Math.round(ev.confidence * 100)}% conf)
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
