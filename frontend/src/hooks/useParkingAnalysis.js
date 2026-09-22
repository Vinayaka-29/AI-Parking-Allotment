import { useCallback, useRef } from 'react'
import { useParkingState } from '../context/ParkingContext'
import { analyzeImageInBrowser } from '../utils/browserVision'

const API_BASE = import.meta.env.VITE_API_BASE || 'https://ai-park-backend.onrender.com/api'

/**
 * Custom hook encapsulating the entire analysis workflow.
 * Manages state transitions: IMAGE_UPLOADED → ANALYZING → ANALYSIS_COMPLETE | ANALYSIS_ERROR
 * Never generates fake parking data.
 */
export function useParkingAnalysis() {
  const { state, dispatch } = useParkingState()
  const abortRef = useRef(null)

  const candidateUrls = [
    'http://127.0.0.1:8000/api',
    API_BASE,
    '/api',
  ]

  /**
   * Upload an image and prepare it for analysis.
   * Does NOT trigger analysis — user must click Analyze.
   */
  const uploadImage = useCallback((file) => {
    if (!file) return
    const previewUrl = URL.createObjectURL(file)
    dispatch({
      type: 'SET_IMAGE_UPLOADED',
      payload: { file, previewUrl },
    })
  }, [dispatch])

  /**
   * Simulate analysis progress steps with delays for UX
   */
  const runProgressSteps = useCallback(async () => {
    const stepDelays = [400, 600, 500, 400, 300]
    for (let i = 0; i < stepDelays.length; i++) {
      await new Promise(r => setTimeout(r, stepDelays[i]))
      dispatch({
        type: 'UPDATE_ANALYSIS_PROGRESS',
        payload: { stepIndex: i, status: 'active' },
      })
    }
  }, [dispatch])

  /**
   * Run analysis on the currently uploaded image.
   * Tries backend first, falls back to browser engine.
   */
  const analyzeImage = useCallback(async (file) => {
    const targetFile = file || state.uploadedFile
    if (!targetFile) return

    dispatch({ type: 'SET_ANALYZING' })

    // Run progress animation concurrently
    const progressPromise = runProgressSteps()

    const formData = new FormData()
    formData.append('file', targetFile)

    let data = null

    // Try backend endpoints
    for (const baseUrl of candidateUrls) {
      if (!baseUrl) continue
      try {
        const controller = new AbortController()
        abortRef.current = controller
        const res = await fetch(`${baseUrl}/detect/image`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        })
        if (res.ok) {
          data = await res.json()
          dispatch({ type: 'SET_BACKEND_CONNECTED', payload: true })
          break
        }
      } catch (e) {
        if (e.name === 'AbortError') return
        // try next endpoint
      }
    }

    // Wait for progress animation to finish
    await progressPromise

    if (data) {
      // Backend returned real analysis
      dispatch({
        type: 'SET_ANALYSIS_COMPLETE',
        payload: data,
      })
    } else {
      // Fall back to browser-side vision engine
      console.warn('Backend unavailable, running browser vision engine...')
      dispatch({ type: 'SET_BACKEND_CONNECTED', payload: false })

      try {
        const browserData = await analyzeImageInBrowser(targetFile)
        dispatch({
          type: 'SET_ANALYSIS_COMPLETE',
          payload: browserData,
        })
      } catch (err) {
        dispatch({
          type: 'SET_ANALYSIS_ERROR',
          payload: { message: `Analysis failed: ${err.message}` },
        })
      }
    }
  }, [state.uploadedFile, dispatch, runProgressSteps])

  /**
   * Upload AND immediately analyze (for camera frames, demos, etc.)
   */
  const uploadAndAnalyze = useCallback(async (file) => {
    if (!file) return
    const previewUrl = URL.createObjectURL(file)
    dispatch({
      type: 'SET_IMAGE_UPLOADED',
      payload: { file, previewUrl },
    })
    // Small delay so the preview renders first
    await new Promise(r => setTimeout(r, 100))
    await analyzeImage(file)
  }, [dispatch, analyzeImage])

  /**
   * Clear all analysis data, return to NO_ANALYSIS state
   */
  const clearAnalysis = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
    }
    if (state.imagePreviewUrl) {
      URL.revokeObjectURL(state.imagePreviewUrl)
    }
    dispatch({ type: 'CLEAR_ANALYSIS' })
  }, [dispatch, state.imagePreviewUrl])

  /**
   * Retry analysis with the same image
   */
  const retryAnalysis = useCallback(() => {
    if (state.uploadedFile) {
      analyzeImage(state.uploadedFile)
    }
  }, [state.uploadedFile, analyzeImage])

  /**
   * Fetch current data from backend (for polling)
   */
  const fetchBackendData = useCallback(async () => {
    try {
      const [ovRes, slRes, evRes] = await Promise.all([
        fetch(`${API_BASE}/overview`).catch(() => null),
        fetch(`${API_BASE}/slots`).catch(() => null),
        fetch(`${API_BASE}/events`).catch(() => null),
      ])

      const payload = {}
      if (ovRes?.ok) payload.overview = await ovRes.json()
      if (slRes?.ok) {
        const slData = await slRes.json()
        if (slData.slots?.length) payload.slots = slData.slots
      }
      if (evRes?.ok) {
        const evData = await evRes.json()
        payload.events = evData.events || []
      }

      if (Object.keys(payload).length > 0) {
        dispatch({ type: 'SET_BACKEND_DATA', payload })
      }
    } catch (err) {
      // Backend unavailable — silent
    }
  }, [dispatch])

  /**
   * Allocate a parking space for a vehicle.
   * Only works when analysis data exists with available slots.
   */
  const allocateSlot = useCallback(async (licensePlate, vehicleType, preferredSlot = null) => {
    const availableSlots = state.slots.filter(s => s.status === 'AVAILABLE')
    if (availableSlots.length === 0) {
      return { error: 'All detected parking spaces are currently occupied.' }
    }

    const plate = licensePlate.trim() || `KA-${Math.floor(Math.random() * 89 + 10)}-GT-${Math.floor(Math.random() * 8999 + 1000)}`

    // Try backend first
    try {
      const res = await fetch(`${API_BASE}/allocate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vehicle_id: plate,
          vehicle_type: vehicleType,
          preferred_slot: preferredSlot?.slot_id || undefined,
        }),
      })
      if (res.ok) {
        const ticketData = await res.json()
        dispatch({ type: 'SET_ISSUED_TICKET', payload: ticketData })
        await fetchBackendData()
        return { success: true, ticket: ticketData }
      }
    } catch (err) {
      // Fallback to local allocation from real detected slots
    }

    // Local allocation from REAL detected slots only (not hardcoded)
    if (availableSlots.length === 0) {
      return { error: 'No available slots detected.' }
    }

    // Pick preferred slot if available, else nearest available slot
    let bestSlot = (preferredSlot && preferredSlot.status === 'AVAILABLE')
      ? preferredSlot
      : null

    if (!bestSlot) {
      const sortedSlots = [...availableSlots].sort(
        (a, b) => (a.distance_from_entries || 999) - (b.distance_from_entries || 999)
      )
      bestSlot = sortedSlots[0]
    }

    // Update slot status
    dispatch({
      type: 'SLOT_STATUS_CHANGED',
      payload: { slot_id: bestSlot.slot_id, status: 'RESERVED', confidence: 0.95 },
    })

    const ticket = {
      ticket_id: `TKT-${Math.floor(Math.random() * 89999 + 10000)}`,
      vehicle_id: plate,
      vehicle_type: vehicleType,
      slot_id: bestSlot.slot_id,
      section_id: bestSlot.section_id || 'A',
      status: 'RESERVED',
      distance: bestSlot.distance_from_entries || 0,
      issued_at: new Date().toISOString(),
    }

    dispatch({ type: 'SET_ISSUED_TICKET', payload: ticket })
    dispatch({
      type: 'ADD_EVENT',
      payload: {
        event_id: `EVT-${Date.now()}`,
        slot_id: bestSlot.slot_id,
        status: 'RESERVED',
        event_type: 'VEHICLE_ASSIGNED',
        timestamp: new Date().toISOString(),
        vehicle_id: plate,
        confidence: 0.95,
      },
    })
    dispatch({
      type: 'SET_STATUS_MSG',
      payload: `SPACE ${bestSlot.slot_id} RESERVED FOR ${plate}`,
    })

    return { success: true, ticket }
  }, [state.slots, dispatch, fetchBackendData])

  /**
   * Re-analyze currently loaded image against a newly calibrated custom layout
   */
  const analyzeWithLayout = useCallback(async (customLayout) => {
    if (!state.uploadedFile && !state.imagePreviewUrl) return
    dispatch({ type: 'SET_ANALYZING' })
    try {
      let fileToAnalyze = state.uploadedFile
      if (!fileToAnalyze && state.imagePreviewUrl) {
        const res = await fetch(state.imagePreviewUrl)
        fileToAnalyze = await res.blob()
      }
      const browserData = await analyzeImageInBrowser(fileToAnalyze, customLayout)
      dispatch({
        type: 'SET_ANALYSIS_COMPLETE',
        payload: browserData,
      })
    } catch (err) {
      dispatch({
        type: 'SET_ANALYSIS_ERROR',
        payload: { message: `Layout analysis error: ${err.message}` },
      })
    }
  }, [state.uploadedFile, state.imagePreviewUrl, dispatch])

  /**
   * Remove a history entry
   */
  const removeHistoryEntry = useCallback((id) => {
    dispatch({ type: 'REMOVE_HISTORY_ENTRY', payload: id })
  }, [dispatch])

  return {
    uploadImage,
    analyzeImage,
    uploadAndAnalyze,
    analyzeWithLayout,
    clearAnalysis,
    retryAnalysis,
    fetchBackendData,
    allocateSlot,
    removeHistoryEntry,
    candidateUrls,
    API_BASE,
  }
}
