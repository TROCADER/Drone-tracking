package dji.sampleV5.aircraft

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.sdk.keyvalue.value.gimbal.CtrlInfo
import dji.sdk.keyvalue.value.gimbal.GimbalSpeedRotation
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.inner.diagnostic.fly.DJIDiagnosticsError.Gimbal
import dji.v5.manager.KeyManager
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import dji.v5.manager.interfaces.ICameraStreamManager.AvailableCameraUpdatedListener
import dji.v5.ux.core.widget.fpv.FPVWidget
import java.nio.ByteBuffer
import java.util.Timer
import java.util.TimerTask
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale


class CameraActivity : AppCompatActivity() {
    private var btnPitchUp: Button? = null;
    private var btnPitchDown: Button? = null;
    private var btnYawLeft: Button? = null;
    private var btnYawRight: Button? = null;
    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private var btnToggleDetection: Switch? = null
    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private var btnToggleTracking: Switch? = null
    private var seekBarConfidence: SeekBar? = null
    private var confidenceLabel: TextView? = null

    private var detectionOverlayView: DetectionOverlay? = null
    private val detector: Detector = Detector()
    private var detectionActive: Boolean = false
    private var trackingActive: Boolean = false

    private val gimbal: Gimbal? = null
    private val mainHandler: Handler = Handler(Looper.getMainLooper())
    private val gimbalTimer: Timer? = null
    private val gimbalTimerTask: TimerTask? = null

    private lateinit var primaryFpvWidget: FPVWidget

    private val availableCameraUpdatedListener = object : AvailableCameraUpdatedListener {
        override fun onAvailableCameraUpdated(availableCameraList: MutableList<ComponentIndexType>) {
            if (availableCameraList.isNotEmpty()) {
                runOnUiThread {
                    // Update stream to the first available camera (e.g. LEFT_OR_MAIN, PORT_1)
                    primaryFpvWidget.updateVideoSource(availableCameraList[0])
                }
            }
        }

        override fun onCameraStreamEnableUpdate(cameraStreamEnableMap: MutableMap<ComponentIndexType, Boolean>) {
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.camera_view)

        detectionOverlayView = findViewById(R.id.detection_overlay)

        Thread {
            detector.loadModel(applicationContext)
        }.start()

        primaryFpvWidget = findViewById(R.id.widget_primary_fpv)
        primaryFpvWidget.isCameraSourceNameVisible = false
        primaryFpvWidget.isCameraSourceSideVisible = false

        primaryFpvWidget.updateVideoSource(ComponentIndexType.LEFT_OR_MAIN)

        MediaDataCenter.getInstance().cameraStreamManager.addAvailableCameraUpdatedListener(availableCameraUpdatedListener)

        btnPitchUp = findViewById(R.id.btn_pitchup)
        btnPitchDown = findViewById(R.id.btn_pitchdown)
        btnYawLeft = findViewById(R.id.btn_yawleft)
        btnYawRight = findViewById(R.id.btn_yawright)
        btnToggleTracking = findViewById(R.id.enable_track)
        btnToggleDetection = findViewById(R.id.enable_det)
        seekBarConfidence = findViewById(R.id.conf_slider)
        confidenceLabel = findViewById(R.id.conf_slider_text)

        setupGimbalButton(btnPitchUp, pitchSpeed = 15.0, yawSpeed = 0.0)
        setupGimbalButton(btnPitchDown, pitchSpeed = -15.0, yawSpeed = 0.0)
        setupGimbalButton(btnYawLeft, pitchSpeed = 0.0, yawSpeed = -15.0)
        setupGimbalButton(btnYawRight, pitchSpeed = 0.0, yawSpeed = 15.0)

        setupToggle(btnToggleDetection) { isChecked ->
            detectionActive = isChecked
            if (isChecked) {
                if (!detector.isModelLoaded()) {
                    Thread {
                        detector.loadModel(applicationContext)
                    }.start()
                }
                startFrameCapture()
            } else {
                stopFrameCapture()
                detectionOverlayView?.clear()
            }
        }

        setupToggle(btnToggleTracking) { isChecked ->
            trackingActive = isChecked
        }
    }

    private val frameListener = ICameraStreamManager.CameraFrameListener { frameData, offset, length, width, height, format ->
        if (!detectionActive)
            return@CameraFrameListener

        val bitmap: Bitmap = createBitmap(width, height)
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(frameData, offset, length))

        val scaledBitmap: Bitmap = bitmap.scale(320, 320)
        bitmap.recycle()

        val result = detector.detect(scaledBitmap)
        scaledBitmap.recycle()

        // TODO: Update PID controller on main thread
        mainHandler.post {
            if (result.success) {
                detectionOverlayView?.setTargets(result.targets)
                Log.d("Detection", "Success: ${result.targets.size} targets")
            } else {
                Log.d("Detection", "Failure: ${result.errorMessage}")
            }
        }
    }

    fun startFrameCapture(cameraIndex: ComponentIndexType = ComponentIndexType.LEFT_OR_MAIN) {
        MediaDataCenter.getInstance().cameraStreamManager.addFrameListener(
            cameraIndex,
            ICameraStreamManager.FrameFormat.RGBA_8888, // Supported: RGBA_8888, NV21, YUV420_888, YUY2
            frameListener
        )
    }

    fun stopFrameCapture() {
        MediaDataCenter.getInstance().cameraStreamManager.removeFrameListener(frameListener)
    }


    @SuppressLint("ClickableViewAccessibility")
    private fun setupGimbalButton(
        button: Button?,
        pitchSpeed: Double,
        yawSpeed: Double
    ) {
        button?.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    rotateGimbalBySpeed(pitchSpeed, yawSpeed)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    rotateGimbalBySpeed(0.0, 0.0)
                    true
                }
                else -> false
            }
        }
    }

    private fun rotateGimbalBySpeed(pitchSpeed: Double, yawSpeed: Double) {
        val rotation = GimbalSpeedRotation(pitchSpeed, yawSpeed, 0.0, CtrlInfo())
        KeyManager.getInstance().performAction(
            KeyTools.createKey(GimbalKey.KeyRotateBySpeed, ComponentIndexType.LEFT_OR_MAIN),
            rotation,
            object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(msg: EmptyMsg?) {}
                override fun onFailure(error: IDJIError) {}
            }
        )
    }

    private fun setupToggle(
        @SuppressLint("UseSwitchCompatOrMaterialCode") switchView: Switch?,
        onToggle: (Boolean) -> Unit
    ) {
        switchView?.setOnCheckedChangeListener { _, isChecked ->
            onToggle(isChecked)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        MediaDataCenter.getInstance().cameraStreamManager.removeAvailableCameraUpdatedListener(availableCameraUpdatedListener)
    }
}