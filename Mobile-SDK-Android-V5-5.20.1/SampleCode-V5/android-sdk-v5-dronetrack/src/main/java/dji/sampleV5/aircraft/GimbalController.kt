package dji.sampleV5.aircraft

import android.util.Log
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.RectF
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class GimbalController(val activity: CameraActivity) {
    private val TAG = "GimbalTrackingController"

    val trackingActive: Boolean
        get() = activity.trackingActive

    val gimbal: ComponentIndexType
        get() = activity.gimbal

    val gimbalRefreshRate: Double = 10.0 // # Hz

    val gimbalMaxDegPitchUp: Double = 80.0
    val gimbalMaxDegPitchDown: Double = -135.0

    val gimbalMaxDegRollLeft: Double = -135.0
    val gimbalMaxDegRollRight: Double = 45.0

    val gimbalMaxDegYawLeft: Double = -30.0
    val gimbalMaxDegYawRight: Double = 30.0

    private var gimbalMaxSpeed: Double = 100.0 // degrees/sec
    private var maxIntegral: Double = 10.0
    private var deadZone: Double = 0.015 // % of screen dimensions

    inner class PidController(
        private var kp: Double,
        private var ki: Double,
        private var kd: Double,
        private var maxOutput: Double,
        private var maxIntegral: Double
    ) {
        private var prevError: Double = 0.0
        private var integral: Double = 0.0
        private var lastTimeMs: Long = 0

        fun setGains(kp: Double, ki: Double, kd: Double) {
            this.kp = kp
            this.ki = ki
            this.kd = kd
        }

        fun setMaxOutput(maxOutput: Double) {
            this.maxOutput = maxOutput
        }

        fun setMaxIntegral(maxIntegral: Double) {
            this.maxIntegral = maxIntegral
        }

        fun compute(error: Double, currentTimeMs: Long): Double {
            var dt: Double = 1.0 / gimbalRefreshRate
            if (lastTimeMs != 0L && currentTimeMs > lastTimeMs) {
                dt = (currentTimeMs - lastTimeMs) / 1000.0
            }
            lastTimeMs = currentTimeMs

            val pTerm = kp * error

            integral += error * dt
            integral = max(-maxIntegral, min(maxIntegral, integral))
            val iTerm = ki * integral

            val derivative = (error - prevError) / dt
            prevError = error
            val dTerm = kd * derivative

            val output = pTerm + iTerm + dTerm
            return max(-maxOutput, min(maxOutput, output))
        }

        fun reset() {
            prevError = 0.0
            integral = 0.0
            lastTimeMs = 0
        }
    }

    private val pidControllerPitch = PidController(100.0, 0.0, 0.0, gimbalMaxSpeed, maxIntegral)
    private val pidControllerRoll = PidController(100.0, 0.0, 0.0, gimbalMaxSpeed, maxIntegral)
    private val pidControllerYaw = PidController(100.0, 0.0, 0.0, gimbalMaxSpeed, maxIntegral)

    var kpPitch: Double = 100.0
        set(value) {
            field = max(0.0, value)
            pidControllerPitch.setGains(field, kiPitch, kdPitch)
        }
    var kiPitch: Double = 0.0
        set(value) {
            field = max(0.0, value)
            pidControllerPitch.setGains(kpPitch, field, kdPitch)
        }
    var kdPitch: Double = 0.0
        set(value) {
            field = max(0.0, value)
            pidControllerPitch.setGains(kpPitch, kiPitch, field)
        }

    var kpRoll: Double = 100.0
        set(value) {
            field = max(0.0, value)
            pidControllerRoll.setGains(field, kiRoll, kdRoll)
        }
    var kiRoll: Double = 0.0
        set(value) {
            field = max(0.0, value)
            pidControllerRoll.setGains(kpRoll, field, kdRoll)
        }
    var kdRoll: Double = 0.0
        set(value) {
            field = max(0.0, value)
            pidControllerRoll.setGains(kpRoll, kiRoll, field)
        }

    var kpYaw: Double = 100.0
        set(value) {
            field = max(0.0, value)
            pidControllerYaw.setGains(field, kiYaw, kdYaw)
        }
    var kiYaw: Double = 0.0
        set(value) {
            field = max(0.0, value)
            pidControllerYaw.setGains(kpYaw, field, kdYaw)
        }
    var kdYaw: Double = 0.0
        set(value) {
            field = max(0.0, value)
            pidControllerYaw.setGains(kpYaw, kiYaw, field)
        }

    private var currentPitch: Double = 0.0
    private var currentYaw: Double = 0.0
    private var currentRoll: Double = 0.0

    private var isGimbalStateValid = true
    private var lastRunTimestampMs: Long = 0L

    fun reset() {
        pidControllerPitch.reset()
        pidControllerRoll.reset()
        pidControllerYaw.reset()
    }

    private fun sendGimbalSpeed(pitchSpeed: Double, yawSpeed: Double, rollSpeed: Double) {
        activity.rotateGimbalBySpeed(pitchSpeed, yawSpeed, rollSpeed)
    }

    @Synchronized
    fun updateTarget(targetBoundingBox: RectF?, timestampMs: Long) {
        if (!trackingActive) {
            return
        }

        // Rate limiting: enforce minimum 100 ms interval (10 Hz refresh rate)
        if (lastRunTimestampMs != 0L) {
            val timeSinceLastRun: Long = timestampMs - lastRunTimestampMs
            val minIntervalMs: Double = 1000.0 / gimbalRefreshRate
            if (timeSinceLastRun < minIntervalMs) {
                return  // Only run PID loop 10 times a second
            }
        }
        lastRunTimestampMs = timestampMs

        if (targetBoundingBox == null) {
            // Target lost: stop gimbal movement and reset PID state
            reset()
            sendGimbalSpeed(0.0, 0.0, 0.0)
            return
        }

        // Calculate center of detected target bounding box in normalized coordinates [0.0, 1.0]
        val left = targetBoundingBox.left ?: 0.0
        val right = targetBoundingBox.right ?: 0.0
        val top = targetBoundingBox.top ?: 0.0
        val bottom = targetBoundingBox.bottom ?: 0.0

        val targetCenterX: Float = ((left + right) / 2.0).toFloat()
        val targetCenterY: Float = ((top + bottom) / 2.0).toFloat()

        // Offset errors relative to frame center (center = 50%
        val errorX = targetCenterX - 0.5
        val errorY = targetCenterY - 0.5

        // Apply deadband (deadzone) filter: if target is within deadzone, stop movement on that axis
        var yawSpeed = 0.0
        if (abs(errorX) < deadZone) {
            pidControllerYaw.reset()
        } else {
            yawSpeed = pidControllerYaw.compute(errorX, timestampMs)
        }

        var pitchSpeed = 0.0
        if (abs(errorY) < deadZone) {
            pidControllerPitch.reset()
        } else {
//            Error down is negative, control signal is positive
            pitchSpeed = pidControllerPitch.compute(-errorY, timestampMs)
        }

        if (isGimbalStateValid) {
            if (currentPitch <= gimbalMaxDegPitchDown && pitchSpeed < 0.0) {
                pitchSpeed = 0.0
                pidControllerPitch.reset()
                Log.w(TAG, "Pitch lower physical limit reached ($currentPitch deg)")
            } else if (currentPitch >= gimbalMaxDegPitchUp && pitchSpeed > 0.0) {
                pitchSpeed = 0.0
                pidControllerPitch.reset()
                Log.w(TAG, "Pitch upper physical limit reached ($currentPitch deg)")
            }

            if (currentYaw <= gimbalMaxDegYawLeft && yawSpeed < 0.0) {
                yawSpeed = 0.0
                pidControllerYaw.reset()
                Log.w(TAG, "Yaw left physical limit reached ($currentYaw deg)")
            } else if (currentYaw >= gimbalMaxDegYawRight && yawSpeed > 0.0) {
                yawSpeed = 0.0
                pidControllerYaw.reset()
                Log.w(TAG, "Yaw right physical limit reached ($currentYaw deg)")
            }
        }

        sendGimbalSpeed(pitchSpeed, yawSpeed, 0.0)
    }
}