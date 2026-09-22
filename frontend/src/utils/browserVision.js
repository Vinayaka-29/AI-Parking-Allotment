/**
 * Client-Side Vision Engine Wrapper.
 * Bridges image uploads/cameras to the autonomous parkingSpaceDetector.
 */
import { analyzeParkingScene, loadCalibratedLayout, saveCalibratedLayout } from './parkingSpaceDetector'

export function analyzeImageInBrowser(imageFile, customLayout = null) {
  return new Promise((resolve, reject) => {
    if (!imageFile) {
      return reject(new Error('No image file provided'))
    }
    const reader = new FileReader()
    reader.onerror = () => reject(new Error('Failed to read image file'))
    reader.onload = (e) => {
      const img = new Image()
      img.onerror = () => reject(new Error('Failed to decode image data'))
      img.onload = () => {
        try {
          const result = analyzeParkingScene(img, customLayout)
          resolve(result)
        } catch (err) {
          reject(err)
        }
      }
      img.src = e.target.result
    }
    reader.readAsDataURL(imageFile)
  })
}

export { loadCalibratedLayout, saveCalibratedLayout }
