import { useEffect, useMemo, useState } from 'react'

const apiBase = 'http://localhost:8000/api'

function App() {
  const [overview, setOverview] = useState({ total_slots: 0, available: 0, occupied: 0, reserved: 0, unknown: 0, occupancy_pct: 0 })
  const [slots, setSlots] = useState([])
  const [events, setEvents] = useState([])

  useEffect(() => {
    const load = async () => {
      const [overviewRes, slotsRes, eventsRes] = await Promise.all([
        fetch(`${apiBase}/overview`),
        fetch(`${apiBase}/slots`),
        fetch(`${apiBase}/events`),
      ])

      setOverview(await overviewRes.json())
      setSlots((await slotsRes.json()).slots)
      setEvents((await eventsRes.json()).events)
    }

    load()
    const timer = setInterval(load, 5000)
    return () => clearInterval(timer)
  }, [])

  const mapRows = useMemo(() => {
    const grouped = {}
    for (const slot of slots) {
      const section = slot.section_id || 'A'
      grouped[section] = grouped[section] || []
      grouped[section].push(slot)
    }
    return Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b))
  }, [slots])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AI parking intelligence</p>
          <h1>AI-PARK Dashboard</h1>
        </div>
        <div className="status-pill">System healthy</div>
      </header>

      <section className="stats-grid">
        <div className="stat-card">
          <span>Total slots</span>
          <strong>{overview.total_slots}</strong>
        </div>
        <div className="stat-card">
          <span>Available</span>
          <strong>{overview.available}</strong>
        </div>
        <div className="stat-card">
          <span>Occupied</span>
          <strong>{overview.occupied}</strong>
        </div>
        <div className="stat-card">
          <span>Reserved</span>
          <strong>{overview.reserved}</strong>
        </div>
        <div className="stat-card">
          <span>Occupancy</span>
          <strong>{overview.occupancy_pct}%</strong>
        </div>
      </section>

      <section className="content-grid">
        <div className="panel live-panel">
          <div className="panel-header">
            <h2>Live camera feed</h2>
          </div>
          <div className="camera-screen">
            <div className="camera-overlay">CAM_01 • phone camera</div>
          </div>
        </div>

        <div className="panel map-panel">
          <div className="panel-header">
            <h2>Parking map</h2>
          </div>
          <div className="parking-map">
            {mapRows.map(([section, sectionSlots]) => (
              <div key={section} className="section-row">
                {sectionSlots.map((slot) => (
                  <div key={slot.slot_id} className={`slot ${slot.status.toLowerCase()}`}>
                    <span>{slot.slot_id}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="bottom-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>Recent events</h2>
          </div>
          <ul className="event-list">
            {events.length
              ? events.slice().reverse().map((event) => (
                  <li key={event.event_id || event.timestamp}>
                    <span>{event.timestamp?.slice(11, 19)}</span>
                    <strong>{event.slot_id}</strong>
                    <em>{event.event_type || event.status}</em>
                  </li>
                ))
              : <li>No events yet</li>}
          </ul>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>System health</h2>
          </div>
          <div className="health-list">
            <div><label>Camera</label><span>ONLINE</span></div>
            <div><label>Detection</label><span>READY</span></div>
            <div><label>Tracking</label><span>STABLE</span></div>
            <div><label>Occupancy</label><span>LIVE</span></div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App
