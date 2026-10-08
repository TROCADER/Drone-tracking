package com.dji.sdk.sample.demo.gimbal;

import android.app.Service;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;

import com.dji.sdk.sample.R;
import com.dji.sdk.sample.internal.controller.DJISampleApplication;
import com.dji.sdk.sample.internal.controller.MainActivity;
import com.dji.sdk.sample.internal.utils.ModuleVerificationUtil;
import com.dji.sdk.sample.internal.utils.OnScreenJoystick;
import com.dji.sdk.sample.internal.utils.ToastUtils;
import com.dji.sdk.sample.internal.utils.VideoFeedView;
import com.dji.sdk.sample.internal.view.PresentableView;

import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

import dji.common.gimbal.Rotation;
import dji.common.gimbal.RotationMode;
import dji.sdk.base.BaseProduct;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.gimbal.Gimbal;
import dji.sdk.products.Aircraft;
import dji.sdk.sdkmanager.DJISDKManager;

/**
 * View presenting a live video feed from the aircraft camera along with gimbal controls,
 * PyTorch Human Detection, and red bounding box overlay targeting.
 */
public class StreamAndGimbalView extends RelativeLayout implements View.OnClickListener, PresentableView {

    private static final String TAG = "StreamAndGimbalView";

    private VideoFeedView primaryVideoFeed;
    private DetectionOverlayView detectionOverlayView;

    // AI Detection UI Elements
    private Button btnToggleDetection;
    private TextView tvDetectionStatus;
    private TextView tvDetectionResult;

    // AI Detection Backend
    private final PyTorchDetector detector = new PyTorchDetector();
    private boolean isDetectionRunning = false;
    private HandlerThread detectionThread;
    private Handler detectionHandler;

    private Gimbal gimbal = null;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Timer gimbalTimer;
    private TimerTask gimbalTimerTask;

    private final Runnable detectionRunnable = new Runnable() {
        @Override
        public void run() {
            if (!isDetectionRunning || primaryVideoFeed == null) {
                return;
            }

            // Capture frame from video feed for SSDLite (320x320)
            final Bitmap frame = primaryVideoFeed.captureFrame(320, 320);
            if (frame != null) {
                final PyTorchDetector.DetectionResult result = detector.detect(frame);
                frame.recycle();

                mainHandler.post(() -> {
                    if (!isDetectionRunning) {
                        return;
                    }
                    if (result.success) {
                        if (detectionOverlayView != null) {
                            detectionOverlayView.setTargets(result.targets);
                        }
                        PyTorchDetector.Recognition top = result.getTopRecognition();
                        if (top != null) {
                            tvDetectionStatus.setText("PyTorch AI: Target Locked");
                            tvDetectionResult.setText(String.format(Locale.US,
                                    "Target: %s (Score: %.1f%%) | Targets: %d | Latency: %d ms",
                                    top.title, top.confidence * 100.0f, result.targets.size(), result.inferenceTimeMs));
                        } else {
                            tvDetectionResult.setText(String.format(Locale.US,
                                    "Searching for human targets | Latency: %d ms", result.inferenceTimeMs));
                        }
                    } else {
                        if (detectionOverlayView != null) {
                            detectionOverlayView.clear();
                        }
                        tvDetectionResult.setText("Error: " + result.errorMessage);
                    }
                });
            } else {
                mainHandler.post(() -> {
                    if (isDetectionRunning && tvDetectionResult != null) {
                        tvDetectionResult.setText("Waiting for video frame...");
                    }
                });
            }

            if (isDetectionRunning && detectionHandler != null) {
                detectionHandler.postDelayed(this, 150); // ~6.6 FPS
            }
        }
    };

    public StreamAndGimbalView(Context context) {
        super(context);
        init(context);
    }

    public StreamAndGimbalView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    private void init(Context context) {
        setClickable(true);
        LayoutInflater layoutInflater = (LayoutInflater) context.getSystemService(Service.LAYOUT_INFLATER_SERVICE);
        layoutInflater.inflate(R.layout.view_stream_and_gimbal, this, true);

        startDetectionThread();
        initUI();
    }

    private void startDetectionThread() {
        detectionThread = new HandlerThread("PyTorchDetectionThread");
        detectionThread.start();
        detectionHandler = new Handler(detectionThread.getLooper());
    }

    private void initUI() {
        primaryVideoFeed = findViewById(R.id.video_feed_view);
        detectionOverlayView = findViewById(R.id.detection_overlay_view);

        View coverView = findViewById(R.id.tv_cover_view);
        if (primaryVideoFeed != null && coverView != null) {
            primaryVideoFeed.setCoverView(coverView);
        }

        btnToggleDetection = findViewById(R.id.btn_toggle_detection);
        tvDetectionStatus = findViewById(R.id.tv_detection_status);
        tvDetectionResult = findViewById(R.id.tv_detection_result);

        Button btnPitchUp = findViewById(R.id.btn_pitch_up);
        Button btnPitchDown = findViewById(R.id.btn_pitch_down);
        Button btnYawLeft = findViewById(R.id.btn_yaw_left);
        Button btnYawRight = findViewById(R.id.btn_yaw_right);
        Button btnResetGimbal = findViewById(R.id.btn_reset_gimbal);

//        Button btnModeFollow = findViewById(R.id.btn_mode_yaw_follow);
//        Button btnModeFpv = findViewById(R.id.btn_mode_fpv);
//        Button btnModeFree = findViewById(R.id.btn_mode_free);

        OnScreenJoystick joystickGimbal = findViewById(R.id.joystick_gimbal);

        if (btnToggleDetection != null) {
            btnToggleDetection.setOnClickListener(this);
        }

        btnPitchUp.setOnClickListener(this);
        btnPitchDown.setOnClickListener(this);
        btnYawLeft.setOnClickListener(this);
        btnYawRight.setOnClickListener(this);
        btnResetGimbal.setOnClickListener(this);
//        btnModeFollow.setOnClickListener(this);
//        btnModeFpv.setOnClickListener(this);
//        btnModeFree.setOnClickListener(this);

        if (joystickGimbal != null) {
            joystickGimbal.setJoystickListener((joystick, pX, pY) -> {
                if (Math.abs(pX) < 0.05f) {
                    pX = 0;
                }
                if (Math.abs(pY) < 0.05f) {
                    pY = 0;
                }
                float maxSpeed = 30.0f; // degrees per second
                float pitchSpeed = pY * maxSpeed;
                float yawSpeed = pX * maxSpeed;

                if (pitchSpeed == 0 && yawSpeed == 0) {
                    stopGimbalTimer();
                    sendGimbalSpeed(0, 0);
                } else {
                    startGimbalRotateTimer(pitchSpeed, yawSpeed);
                }
            });
        }
    }

    private void toggleDetection() {
        if (!isDetectionRunning) {
            isDetectionRunning = true;
            if (btnToggleDetection != null) {
                btnToggleDetection.setText("Stop AI Mode");
            }
            if (tvDetectionStatus != null) {
                tvDetectionStatus.setText("PyTorch AI: Loading Model...");
            }
            if (tvDetectionResult != null) {
                tvDetectionResult.setText("Initializing model from assets...");
            }

            detectionHandler.post(() -> {
                final boolean loaded = detector.loadModel(getContext());
                mainHandler.post(() -> {
                    if (!isDetectionRunning) {
                        return;
                    }
                    if (loaded) {
                        tvDetectionStatus.setText("PyTorch AI: Active");
                        tvDetectionResult.setText("Scanning video feed for human targets...");
                        ToastUtils.setResultToToast("PyTorch AI model loaded!");
                        detectionHandler.post(detectionRunnable);
                    } else {
                        isDetectionRunning = false;
                        if (btnToggleDetection != null) {
                            btnToggleDetection.setText("Start AI Mode");
                        }
                        if (detectionOverlayView != null) {
                            detectionOverlayView.clear();
                        }
                        tvDetectionStatus.setText("PyTorch AI: Failed");
                        tvDetectionResult.setText("Failed to load model from assets");
                        ToastUtils.setResultToToast("Failed to load PyTorch model");
                    }
                });
            });
        } else {
            isDetectionRunning = false;
            if (detectionHandler != null) {
                detectionHandler.removeCallbacks(detectionRunnable);
            }
            if (detectionOverlayView != null) {
                detectionOverlayView.clear();
            }
            if (btnToggleDetection != null) {
                btnToggleDetection.setText("Start AI Mode");
            }
            if (tvDetectionStatus != null) {
                tvDetectionStatus.setText("PyTorch AI: Off");
            }
            if (tvDetectionResult != null) {
                tvDetectionResult.setText("Result: Click 'Start AI Mode' to run human detection");
            }
            ToastUtils.setResultToToast("PyTorch AI Detection stopped");
        }
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        DJISampleApplication.getEventBus().post(new MainActivity.RequestStartFullScreenEvent());

        if (primaryVideoFeed != null && VideoFeeder.getInstance() != null) {
            VideoFeeder.getInstance().getPrimaryVideoFeed();
            primaryVideoFeed.registerLiveVideo(VideoFeeder.getInstance().getPrimaryVideoFeed(), true);
        }

        initGimbalStateListener();
    }

    @Override
    protected void onDetachedFromWindow() {
        DJISampleApplication.getEventBus().post(new MainActivity.RequestEndFullScreenEvent());
        stopGimbalTimer();
        removeGimbalStateListener();

        isDetectionRunning = false;
        if (detectionHandler != null) {
            detectionHandler.removeCallbacksAndMessages(null);
        }
        if (detectionOverlayView != null) {
            detectionOverlayView.clear();
        }
        if (detectionThread != null) {
            detectionThread.quitSafely();
        }

        super.onDetachedFromWindow();
    }

    private Gimbal getGimbal() {
        if (gimbal == null) {
            if (ModuleVerificationUtil.isGimbalModuleAvailable()) {
                gimbal = DJISampleApplication.getProductInstance().getGimbal();
            } else if (DJISDKManager.getInstance() != null && DJISDKManager.getInstance().getProduct() != null) {
                BaseProduct product = DJISDKManager.getInstance().getProduct();
                if (product instanceof Aircraft aircraft) {
                    if (aircraft.getGimbals() != null && !aircraft.getGimbals().isEmpty()) {
                        gimbal = aircraft.getGimbals().get(0);
                    }
                } else {
                    gimbal = product.getGimbal();
                }
            }
        }
        return gimbal;
    }

    private void initGimbalStateListener() {
        Gimbal g = getGimbal();
        if (g != null) {
            g.setStateCallback(gimbalState -> mainHandler.post(() -> {
                // Gimbal state listener callback
            }));
        }
    }

    private void removeGimbalStateListener() {
        Gimbal g = getGimbal();
        if (g != null) {
            g.setStateCallback(null);
        }
    }

    @Override
    public void onClick(View v) {
        int id = v.getId();
        if (id == R.id.btn_toggle_detection) {
            toggleDetection();
        } else if (id == R.id.btn_pitch_up) {
            rotateGimbalRelative(10.0f, 0);
        } else if (id == R.id.btn_pitch_down) {
            rotateGimbalRelative(-10.0f, 0);
        } else if (id == R.id.btn_yaw_left) {
            rotateGimbalRelative(0, -10.0f);
        } else if (id == R.id.btn_yaw_right) {
            rotateGimbalRelative(0, 10.0f);
        } else if (id == R.id.btn_reset_gimbal) {
            resetGimbal();
        }
//        } else if (id == R.id.btn_mode_yaw_follow) {
//            setGimbalMode(GimbalMode.YAW_FOLLOW);
//        } else if (id == R.id.btn_mode_fpv) {
//            setGimbalMode(GimbalMode.FPV);
//        } else if (id == R.id.btn_mode_free) {
//            setGimbalMode(GimbalMode.FREE);
//        }
    }

    private void rotateGimbalRelative(float pitchAngle, float yawAngle) {
        Gimbal g = getGimbal();
        if (g != null) {
            Rotation rotation = new Rotation.Builder()
                    .pitch(pitchAngle)
                    .yaw(yawAngle)
                    .roll(0)
                    .mode(RotationMode.RELATIVE_ANGLE)
                    .time(0.5)
                    .build();
            g.rotate(rotation, error -> {
                if (error != null) {
                    ToastUtils.setResultToToast("Gimbal error: " + error.getDescription());
                }
            });
        } else {
            ToastUtils.setResultToToast("Gimbal disconnected!");
        }
    }

    private void resetGimbal() {
        Gimbal g = getGimbal();
        if (g != null) {
            g.reset(error -> {
                if (error == null) {
                    ToastUtils.setResultToToast("Gimbal Reset Succeeded");
                } else {
                    ToastUtils.setResultToToast("Gimbal Reset Failed: " + error.getDescription());
                }
            });
        } else {
            ToastUtils.setResultToToast("Gimbal disconnected!");
        }
    }

//    private void setGimbalMode(final GimbalMode mode) {
//        Gimbal g = getGimbal();
//        if (g != null) {
//            g.setMode(mode, error -> {
//                if (error == null) {
//                    ToastUtils.setResultToToast("Gimbal Mode set to " + mode.name());
//                } else {
//                    ToastUtils.setResultToToast("Set mode failed: " + error.getDescription());
//                }
//            });
//        } else {
//            ToastUtils.setResultToToast("Gimbal disconnected!");
//        }
//    }

    private synchronized void startGimbalRotateTimer(final float pitchSpeed, final float yawSpeed) {
        stopGimbalTimer();
        if (pitchSpeed == 0 && yawSpeed == 0) {
            sendGimbalSpeed(0, 0);
            return;
        }
        gimbalTimer = new Timer();
        gimbalTimerTask = new TimerTask() {
            @Override
            public void run() {
                sendGimbalSpeed(pitchSpeed, yawSpeed);
            }
        };
        gimbalTimer.schedule(gimbalTimerTask, 0, 100);
    }

    private void stopGimbalTimer() {
        if (gimbalTimerTask != null) {
            gimbalTimerTask.cancel();
            gimbalTimerTask = null;
        }
        if (gimbalTimer != null) {
            gimbalTimer.cancel();
            gimbalTimer.purge();
            gimbalTimer = null;
        }
    }

    private void sendGimbalSpeed(float pitchSpeed, float yawSpeed) {
        Gimbal g = getGimbal();
        if (g != null) {
            Rotation rotation = new Rotation.Builder()
                    .pitch(pitchSpeed)
                    .yaw(yawSpeed)
                    .roll(0)
                    .mode(RotationMode.SPEED)
                    .time(0)
                    .build();
            g.rotate(rotation, error -> {
                if (error != null) {
                    Log.e(TAG, "Gimbal rotation error: " + error.getDescription());
                }
            });
        }
    }

    @Override
    public int getDescription() {
        return R.string.video_feed_and_gimbal_description;
    }

    @NonNull
    @Override
    public String getHint() {
        return this.getClass().getSimpleName() + ".java";
    }
}
