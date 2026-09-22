import { useEffect, useRef, useCallback } from 'react'
import { useParkingState } from '../context/ParkingContext'

const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://127.0.0.1:8000/api/ws'

/**
 * WebSocket hook for real-time parking updates.
 * Handles SLOT_STATUS_CHANGED, VEHICLE_DETECTED, VEHICLE_ENTERED, VEHICLE_LEFT events.
 * Only updates affected components — no full re-render.
 */
export function useWebSocket() {
  const { dispatch } = useParkingState()
  const wsRef = useRef(null)
  const reconnectTimerRef = useRef(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(WS_BASE)

      ws.onopen = () => {
        console.log('WebSocket connected')
        dispatch({ type: 'SET_WS_CONNECTED', payload: true })
        reconnectAttemptsRef.current = 0
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          switch (data.type) {
            case 'SLOT_STATUS_CHANGED':
              dispatch({
                type: 'SLOT_STATUS_CHANGED',
                payload: {
                  slot_id: data.slot_id,
                  status: data.status,
                  confidence: data.confidence,
                },
              })
              dispatch({
                type: 'ADD_EVENT',
                payload: {
                  event_id: `WS-${Date.now()}`,
                  slot_id: data.slot_id,
                  status: data.status,
                  event_type: 'SLOT_STATUS_CHANGED',
                  timestamp: new Date().toISOString(),
                  confidence: data.confidence,
                },
              })
              break

            case 'VEHICLE_DETECTED':
            case 'VEHICLE_ENTERED':
            case 'VEHICLE_LEFT':
              dispatch({
                type: 'ADD_EVENT',
                payload: {
                  event_id: `WS-${Date.now()}`,
                  event_type: data.type,
                  timestamp: new Date().toISOString(),
                  vehicle_id: data.vehicle_id,
                  slot_id: data.slot_id,
                  status: data.status,
                  confidence: data.confidence,
                },
              })
              break

            case 'ALLOCATION_CREATED':
            case 'RESERVATION_CREATED':
              dispatch({
                type: 'ADD_EVENT',
                payload: {
                  event_id: `WS-${Date.now()}`,
                  event_type: data.type,
                  timestamp: new Date().toISOString(),
                  vehicle_id: data.vehicle_id,
                  slot_id: data.slot_id,
                  status: data.status || 'RESERVED',
                },
              })
              if (data.slot_id) {
                dispatch({
                  type: 'SLOT_STATUS_CHANGED',
                  payload: {
                    slot_id: data.slot_id,
                    status: 'RESERVED',
                    confidence: data.confidence || 0.95,
                  },
                })
              }
              break

            default:
              break
          }
        } catch (err) {
          console.warn('WebSocket message parse error:', err)
        }
      }

      ws.onclose = () => {
        dispatch({ type: 'SET_WS_CONNECTED', payload: false })

        // Attempt reconnection with exponential backoff
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(2000 * Math.pow(2, reconnectAttemptsRef.current), 30000)
          reconnectAttemptsRef.current++
          reconnectTimerRef.current = setTimeout(connect, delay)
        }
      }

      ws.onerror = () => {
        // Will trigger onclose
      }

      wsRef.current = ws
    } catch (err) {
      // WebSocket not available
    }
  }, [dispatch])

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    dispatch({ type: 'SET_WS_CONNECTED', payload: false })
  }, [dispatch])

  useEffect(() => {
    connect()
    return disconnect
  }, [connect, disconnect])

  return {
    connected: wsRef.current?.readyState === WebSocket.OPEN,
    connect,
    disconnect,
  }
}
