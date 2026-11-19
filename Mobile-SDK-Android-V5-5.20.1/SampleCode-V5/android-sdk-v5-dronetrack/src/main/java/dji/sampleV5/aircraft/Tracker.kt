package dji.sampleV5.aircraft

import dji.sampleV5.aircraft.Detector.TargetBox
import dji.sdk.keyvalue.value.common.RectF
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get

class Tracker {
    private var kFilter: Kalman? = null
    private var missedFrames = 0
    private val maxMissedFrames = 15
    
    // Approximate Camera FOV in degrees (can be tuned for specific drone model)
    private val horizontalFov = 80.0
    private val verticalFov = 45.0
    private var lastTimeMs = 0L

    fun track(
        targets: List<TargetBox>, 
        gimbalPitchSpeed: Double = 0.0, 
        gimbalYawSpeed: Double = 0.0
    ): TargetBox? {
        val currentTimeMs = System.currentTimeMillis()
        // If first frame, assume default dt of 0.1s
        val actualDt = if (lastTimeMs > 0) (currentTimeMs - lastTimeMs) / 1000.0 else 0.1
        lastTimeMs = currentTimeMs

        if (kFilter == null) {
            // Initialize if we have targets, otherwise return null
            if (targets.isNotEmpty()) {
                val initialTarget = targets.maxByOrNull { it.score } ?: targets.first()
                kFilter = Kalman(initialTarget.rect)
                missedFrames = 0
                return initialTarget
            }
            return null
        }
        
        val kalmanFilter = kFilter!!
        
        // Calculate control input u based on gimbal velocities
        // The Kalman filter uses a hardcoded dt=0.1. To apply the correct physical translation,
        // we scale the velocity based on the ratio of actual_dt to 0.1.
        val dtRatio = actualDt / 0.1
        val u = if (gimbalPitchSpeed != 0.0 || gimbalYawSpeed != 0.0) {
            val topVelocity = (gimbalPitchSpeed / verticalFov) * dtRatio
            val leftVelocity = (-gimbalYawSpeed / horizontalFov) * dtRatio
            mk.ndarray(mk[topVelocity, leftVelocity])
        } else {
            null
        }

        kalmanFilter.predict(u)
        val trackedRect = kalmanFilter.currentEstimateRect()

        val selectedBox = targets.maxByOrNull { calculateIoU(it.rect, trackedRect) }
        val maxIou = selectedBox?.let { calculateIoU(it.rect, trackedRect) } ?: 0f

        // Only update if we have a match with non-zero IoU
        if (selectedBox != null && maxIou > 0f) {
            kalmanFilter.update(
                mk.ndarray(
                    mk[
                        selectedBox.rect.top ?: 0.0,
                        selectedBox.rect.left ?: 0.0,
                        (selectedBox.rect.right ?: 0.0) - (selectedBox.rect.left ?: 0.0),
                        (selectedBox.rect.bottom ?: 0.0) - (selectedBox.rect.top ?: 0.0)
                    ]
                )
            )
            missedFrames = 0
            return TargetBox(kalmanFilter.currentEstimateRect(), selectedBox.label, selectedBox.score)
        } else {
            missedFrames++
            if (missedFrames > maxMissedFrames) {
                reset()
                return null
            }
            // Return predicted box with default values if missed but not yet reset
            return TargetBox(kalmanFilter.currentEstimateRect(), "Target", 1.0f)
        }
    }

    fun reset() {
        kFilter = null
        missedFrames = 0
        lastTimeMs = 0L
    }
}