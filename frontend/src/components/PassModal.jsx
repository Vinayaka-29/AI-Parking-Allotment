import React, { useEffect } from 'react'
import confetti from 'canvas-confetti'

export default function PassModal({ ticket, onClose }) {
  useEffect(() => {
    if (ticket) {
      confetti({
        particleCount: 70,
        spread: 60,
        origin: { y: 0.6 },
        colors: ['#00f0ff', '#00ff9d', '#ff0055', '#ffb700'],
      })
    }
  }, [ticket])

  if (!ticket) return null

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-cyber-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="hud-tag">AI TICKET ISSUED</div>
          <button className="modal-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="ticket-body">
          <div className="ticket-badge-banner">
            <span className="car-icon">🏎️</span>
            <div>
              <h3>RESERVATION CONFIRMED</h3>
              <p>Autonomous Parking Guidance</p>
            </div>
          </div>

          <div className="slot-hero-box">
            <span className="slot-label">ASSIGNED BAY</span>
            <span className="slot-number">{ticket.slot_id}</span>
            <span className="slot-section">SECTION {ticket.section_id || 'A'} • NEAREST ENTRY</span>
          </div>

          <div className="ticket-details-grid">
            <div className="ticket-row">
              <span>TICKET ID</span>
              <strong>{ticket.ticket_id || 'TKT-99281'}</strong>
            </div>
            <div className="ticket-row">
              <span>VEHICLE PLATE</span>
              <strong>{ticket.vehicle_id}</strong>
            </div>
            <div className="ticket-row">
              <span>VEHICLE TYPE</span>
              <strong>{ticket.vehicle_type?.toUpperCase() || 'LUXURY CAR'}</strong>
            </div>
            <div className="ticket-row">
              <span>DISTANCE</span>
              <strong>{ticket.distance} METERS</strong>
            </div>
            <div className="ticket-row">
              <span>ISSUED TIME</span>
              <strong>{new Date(ticket.issued_at).toLocaleTimeString()}</strong>
            </div>
          </div>

          {/* Holographic QR Code Simulation */}
          <div className="qr-box">
            <div className="qr-pattern">
              <div className="qr-corner top-left" />
              <div className="qr-corner top-right" />
              <div className="qr-corner bottom-left" />
              <div className="qr-core">
                <span>AI-PASS</span>
                <code>{ticket.slot_id}</code>
              </div>
            </div>
            <p className="qr-caption">SCAN AT BARRIER GATE FOR AUTO-ENTRY</p>
          </div>
        </div>

        <div className="modal-footer">
          <button className="cyber-btn primary" onClick={onClose}>
            CONFIRM & NAVIGATE ➔
          </button>
        </div>
      </div>
    </div>
  )
}
