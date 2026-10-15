package com.dji.sdk.sample.demo.gimbal;

import android.graphics.RectF;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import dji.common.gimbal.GimbalState;
import dji.common.gimbal.Rotation;
import dji.common.gimbal.RotationMode;
import dji.sdk.gimbal.Gimbal;

/**
 * Controller class that uses a Proportional-Integral-Derivative (PID) control loop
 * to adjust gimbal pitch and yaw speed, centering the camera around a detected target bounding box.
 * Takes into account the physical position limits and refresh rate (10 Hz) of the gimbal.
 */
public class GimbalTrackingController {

    private static final String TAG = "GimbalTrackingController";

    // Physical Position Limits (in degrees) and Refresh Rate (in Hz)
    public static float GIMBAL_REFRESH_RATE = 10.0f; // 10 Hz refresh rate (new information 10 times a second)
    public static float MAX_TILT_DEG_UP = 30.0f;     // Physical pitch upper limit
    public static float MAX_TILT_DEG_DOWN = -90.0f;  // Physical pitch lower limit
    public static float MAX_PAN_DEG_LEFT = -75.0f;   // Physical yaw left limit
    public static float MAX_PAN_DEG_RIGHT = 75.0f;   // Physical yaw right limit

    // Default PID Parameters
    private static final float DEFAULT_KP_YAW = 100.0f;
    private static final float DEFAULT_KI_YAW = 0.0f;
    private static final float DEFAULT_KD_YAW = 0.0f;

    private static final float DEFAULT_KP_PITCH = 100.0f;
    private static final float DEFAULT_KI_PITCH = 0.0f;
    private static final float DEFAULT_KD_PITCH = 0.0f;

    private static final float DEFAULT_MAX_SPEED = 120.0f; // degrees/sec
    private static final float DEFAULT_MAX_INTEGRAL = 10.0f;
    private static final float DEFAULT_DEADBAND = 0.015f; // 1.5% of screen dimensions

    /**
     * Inner class representing a single-axis PID controller.
     */
    public static class PidController {
        private float kp;
        private float ki;
        private float kd;
        private float maxOutput;
        private float maxIntegral;

        private float prevError = 0.0f;
        private float integral = 0.0f;
        private long lastTimeMs = 0;

        public PidController(float kp, float ki, float kd, float maxOutput, float maxIntegral) {
            this.kp = kp;
            this.ki = ki;
            this.kd = kd;
            this.maxOutput = maxOutput;
            this.maxIntegral = maxIntegral;
        }

        public void setGains(float kp, float ki, float kd) {
            this.kp = kp;
            this.ki = ki;
            this.kd = kd;
        }

        public void setMaxOutput(float maxOutput) {
            this.maxOutput = maxOutput;
        }

        public void setMaxIntegral(float maxIntegral) {
            this.maxIntegral = maxIntegral;
        }

        /**
         * Calculates PID output for the given error.
         *
         * @param error         The process error (desired - actual).
         * @param currentTimeMs Current timestamp in milliseconds.
         * @return Calculated control output bounded by maxOutput.
         */
        public float compute(float error, long currentTimeMs) {
            float dt = 1.0f / GIMBAL_REFRESH_RATE; // 0.1s for 10 Hz refresh rate
            if (lastTimeMs != 0 && currentTimeMs > lastTimeMs) {
                dt = (currentTimeMs - lastTimeMs) / 1000.0f;
            }
            lastTimeMs = currentTimeMs;

            // Proportional term
            float pTerm = kp * error;

            // Integral term with anti-windup clamping
            integral += error * dt;
            integral = Math.max(-maxIntegral, Math.min(maxIntegral, integral));
            float iTerm = ki * integral;

            // Derivative term
            float derivative = (error - prevError) / dt;
            prevError = error;
            float dTerm = kd * derivative;

            // Total output bounded by maxOutput
            float output = pTerm + iTerm + dTerm;
            return Math.max(-maxOutput, Math.min(maxOutput, output));
        }

        /**
         * Resets the internal controller state (integral accumulators and timestamps).
         */
        public void reset() {
            prevError = 0.0f;
            integral = 0.0f;
            lastTimeMs = 0;
        }
    }

    private final PidController yawPid; // Pan
    private final PidController pitchPid; // Tilt

    private Gimbal gimbal;
    private boolean isTrackingEnabled = false;
    private float deadband = DEFAULT_DEADBAND;

    // Current retrieved gimbal attitude position (in degrees)
    private float currentPitch = 0.0f;
    private float currentYaw = 0.0f;
    private float currentRoll = 0.0f;
    private boolean isGimbalStateValid = false;

    // Rate limiting execution timestamp
    private long lastRunTimestampMs = 0;

    private final GimbalState.Callback gimbalStateCallback = new GimbalState.Callback() {
        @Override
        public void onUpdate(@NonNull GimbalState gimbalState) {
            if (gimbalState.getAttitudeInDegrees() != null) {
                currentPitch = gimbalState.getAttitudeInDegrees().getPitch();
                currentYaw = gimbalState.getAttitudeInDegrees().getYaw();
                currentRoll = gimbalState.getAttitudeInDegrees().getRoll();
                isGimbalStateValid = true;
            }
        }
    };

    public GimbalTrackingController() {
        this(null);
    }

    public GimbalTrackingController(@Nullable Gimbal gimbal) {
        this.yawPid = new PidController(DEFAULT_KP_YAW, DEFAULT_KI_YAW, DEFAULT_KD_YAW,
                DEFAULT_MAX_SPEED, DEFAULT_MAX_INTEGRAL);
        this.pitchPid = new PidController(DEFAULT_KP_PITCH, DEFAULT_KI_PITCH, DEFAULT_KD_PITCH,
                DEFAULT_MAX_SPEED, DEFAULT_MAX_INTEGRAL);
        setGimbal(gimbal);
    }

    public synchronized void setGimbal(@Nullable Gimbal newGimbal) {
        if (this.gimbal != newGimbal) {
            if (this.gimbal != null) {
                this.gimbal.setStateCallback(null);
            }
            this.gimbal = newGimbal;
            if (this.gimbal != null && isTrackingEnabled) {
                this.gimbal.setStateCallback(gimbalStateCallback);
            }
        }
    }

    public synchronized void setTrackingEnabled(boolean enabled) {
        if (this.isTrackingEnabled != enabled) {
            this.isTrackingEnabled = enabled;
            if (gimbal != null) {
                gimbal.setStateCallback(enabled ? gimbalStateCallback : null);
            }
            if (!enabled) {
                stopTrackingAndReset();
            } else {
                reset();
                Log.i(TAG, "Gimbal PID target tracking enabled (10 Hz rate limit)");
            }
        }
    }

    public synchronized boolean isTrackingEnabled() {
        return isTrackingEnabled;
    }

    public void setDeadband(float deadband) {
        this.deadband = deadband;
    }

    public void setYawPidGains(float kp, float ki, float kd) {
        this.yawPid.setGains(kp, ki, kd);
    }

    public void setPitchPidGains(float kp, float ki, float kd) {
        this.pitchPid.setGains(kp, ki, kd);
    }

    public void setMaxSpeed(float maxSpeed) {
        this.yawPid.setMaxOutput(maxSpeed);
        this.pitchPid.setMaxOutput(maxSpeed);
    }

    public float getCurrentPitch() {
        return currentPitch;
    }

    public float getCurrentYaw() {
        return currentYaw;
    }

    public float getCurrentRoll() {
        return currentRoll;
    }

    public boolean isGimbalStateValid() {
        return isGimbalStateValid;
    }

    public GimbalState.Callback getGimbalStateCallback() {
        return gimbalStateCallback;
    }

    /**
     * Resets the PID controllers' error history and timestamp tracker.
     */
    public synchronized void reset() {
        yawPid.reset();
        pitchPid.reset();
        lastRunTimestampMs = 0;
    }

    /**
     * Stops gimbal movement and resets internal PID states.
     */
    public synchronized void stopTrackingAndReset() {
        reset();
        sendGimbalSpeed(0.0f, 0.0f);
        Log.i(TAG, "Gimbal tracking stopped and PID reset");
    }

    /**
     * Process a detected bounding box and update the gimbal position using PID control.
     *
     * @param targetBoundingBox Normalized bounding box (0.0 to 1.0) of target, or null if lost.
     */
    public void updateTarget(@Nullable RectF targetBoundingBox) {
        updateTarget(targetBoundingBox, System.currentTimeMillis());
    }

    /**
     * Process a detected bounding box and update the gimbal position using PID control.
     * Rate-limited to 10 Hz refresh rate (10 times a second).
     *
     * @param targetBoundingBox Normalized bounding box (0.0 to 1.0) of target, or null if lost.
     * @param timestampMs       Timestamp of detection frame in milliseconds.
     */
    public synchronized void updateTarget(@Nullable RectF targetBoundingBox, long timestampMs) {
        if (!isTrackingEnabled) {
            return;
        }

        // Rate limiting: enforce minimum 100 ms interval (10 Hz refresh rate)
        if (lastRunTimestampMs != 0) {
            long timeSinceLastRun = timestampMs - lastRunTimestampMs;
            float minIntervalMs = 1000.0f / GIMBAL_REFRESH_RATE;
            if (timeSinceLastRun < minIntervalMs) {
                return; // Only run PID loop 10 times a second
            }
        }
        lastRunTimestampMs = timestampMs;

        if (targetBoundingBox == null) {
            // Target lost: stop gimbal movement and reset PID state
            reset();
            sendGimbalSpeed(0.0f, 0.0f);
            return;
        }

        // Calculate center of detected target bounding box in normalized coordinates [0.0, 1.0]
        float targetCenterX = targetBoundingBox.centerX();
        float targetCenterY = targetBoundingBox.centerY();

        // Frame center in normalized coordinates
        float frameCenterX = 0.5f;
        float frameCenterY = 0.5f;

        // Offset errors relative to frame center
        float errorX = targetCenterX - frameCenterX;
        float errorY = targetCenterY - frameCenterY;

        // Apply deadband (deadzone) filter: if target is within deadzone, stop movement on that axis
        float yawSpeed = 0.0f;
        if (Math.abs(errorX) < deadband) {
            yawPid.reset();
        } else {
            yawSpeed = yawPid.compute(errorX, timestampMs);
        }

        float pitchSpeed = 0.0f;
        if (Math.abs(errorY) < deadband) {
            pitchPid.reset();
        } else {
            // Invert pitch error because positive pitch speed tilts UP, but positive errorY means target is DOWN
            pitchSpeed = pitchPid.compute(-errorY, timestampMs);
        }

        // Apply physical position limits of gimbal hardware
        if (isGimbalStateValid) {
            // Pitch / Tilt limit checks
            if (currentPitch <= MAX_TILT_DEG_DOWN && pitchSpeed < 0.0f) {
                pitchSpeed = 0.0f;
                pitchPid.reset();
                Log.w(TAG, "Pitch lower physical limit reached (" + currentPitch + " deg)");
            } else if (currentPitch >= MAX_TILT_DEG_UP && pitchSpeed > 0.0f) {
                pitchSpeed = 0.0f;
                pitchPid.reset();
                Log.w(TAG, "Pitch upper physical limit reached (" + currentPitch + " deg)");
            }

            // Yaw / Pan limit checks
            if (currentYaw <= MAX_PAN_DEG_LEFT && yawSpeed < 0.0f) {
                yawSpeed = 0.0f;
                yawPid.reset();
                Log.w(TAG, "Yaw left physical limit reached (" + currentYaw + " deg)");
            } else if (currentYaw >= MAX_PAN_DEG_RIGHT && yawSpeed > 0.0f) {
                yawSpeed = 0.0f;
                yawPid.reset();
                Log.w(TAG, "Yaw right physical limit reached (" + currentYaw + " deg)");
            }
        }

        sendGimbalSpeed(pitchSpeed, yawSpeed);
    }

    /**
     * Sends rotational speed command to the gimbal hardware.
     */
    private void sendGimbalSpeed(float pitchSpeed, float yawSpeed) {
        if (gimbal == null) {
            return;
        }

        Rotation rotation = new Rotation.Builder()
                .pitch(pitchSpeed)
                .yaw(yawSpeed)
                .roll(0)
                .mode(RotationMode.SPEED)
                .time(0)
                .build();

        gimbal.rotate(rotation, error -> {
            if (error != null) {
                Log.e(TAG, "Gimbal rotation speed command failed: " + error.getDescription());
            }
        });
    }
}
