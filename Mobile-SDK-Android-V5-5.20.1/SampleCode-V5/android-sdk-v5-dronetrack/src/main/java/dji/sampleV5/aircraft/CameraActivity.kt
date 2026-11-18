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
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.sdk.keyvalue.value.gimbal.CtrlInfo
import dji.sdk.keyvalue.value.gimbal.GimbalSpeedRotation
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import dji.v5.manager.KeyManager
import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.interfaces.ICameraStreamManager
import dji.v5.manager.interfaces.ICameraStreamManager.AvailableCameraUpdatedListener
import dji.sampleV5.aircraft.models.BasicAircraftControlVM
import dji.v5.ux.core.widget.fpv.FPVWidget
import dji.v5.ux.util.ToastUtils
import java.nio.ByteBuffer
import java.util.Locale


class CameraActivity : AppCompatActivity() {
    private val basicAircraftControlVM = BasicAircraftControlVM()
    private var btnTakeOff: Button? = null
    private var btnLand: Button? = null
    private var btnPitchUp: Button? = null
    private var btnPitchDown: Button? = null
    private var btnYawLeft: Button? = null
    private var btnYawRight: Button? = null

    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private var swtichToggleDetection: Switch? = null

    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private var switchToggleTracking: Switch? = null
    private var tvDetTime: TextView? = null
    private var detTime: Long = 0L

    private var btnPidPitchKpIncrease: Button? = null;
    private var btnPidPitchKpDecrease: Button? = null;
    private var btnPidPitchKiIncrease: Button? = null;
    private var btnPidPitchKiDecrease: Button? = null;
    private var btnPidPitchKdIncrease: Button? = null;
    private var btnPidPitchKdDecrease: Button? = null;

    private var btnPidRollKpIncrease: Button? = null;
    private var btnPidRollKpDecrease: Button? = null;
    private var btnPidRollKiIncrease: Button? = null;
    private var btnPidRollKiDecrease: Button? = null;
    private var btnPidRollKdIncrease: Button? = null;
    private var btnPidRollKdDecrease: Button? = null;

    private var btnPidYawKpIncrease: Button? = null;
    private var btnPidYawKpDecrease: Button? = null;
    private var btnPidYawKiIncrease: Button? = null;
    private var btnPidYawKiDecrease: Button? = null;
    private var btnPidYawKdIncrease: Button? = null;
    private var btnPidYawKdDecrease: Button? = null;

    private var tvPidPitchKp: TextView? = null
    private var tvPidPitchKi: TextView? = null
    private var tvPidPitchKd: TextView? = null

    private var tvPidRollKp: TextView? = null
    private var tvPidRollKi: TextView? = null
    private var tvPidRollKd: TextView? = null

    private var tvPidYawKp: TextView? = null
    private var tvPidYawKi: TextView? = null
    private var tvPidYawKd: TextView? = null

    private var tvConfidence: TextView? = null
    private var tvIou: TextView? = null
    private var seekBarConfidence: SeekBar? = null
    private var seekBarIou: SeekBar? = null

    private var detectionOverlayView: DetectionOverlay? = null
    private var detector: Detector = Detector()
    private var tracker: Tracker = Tracker()
    private var gimbalController: GimbalController? = null
    private var detectionActive: Boolean = false
    var trackingActive: Boolean = false

    val gimbal: ComponentIndexType = ComponentIndexType.LEFT_OR_MAIN
    private val mainHandler: Handler = Handler(Looper.getMainLooper())

    private lateinit var primaryFpvWidget: FPVWidget

    private val availableCameraUpdatedListener = object : AvailableCameraUpdatedListener {
        override fun onAvailableCameraUpdated(availableCameraList: MutableList<ComponentIndexType>) {
            if (availableCameraList.isNotEmpty()) {
                runOnUiThread {
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

        btnTakeOff = findViewById(R.id.btn_takeoff)
        btnLand = findViewById(R.id.btn_land)

        // Takeoff Button Click Listener
        btnTakeOff?.setOnClickListener {
            basicAircraftControlVM.startTakeOff(object :
                CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(msg: EmptyMsg?) {
                    ToastUtils.showToast("Takeoff Succeeded")
                }

                override fun onFailure(error: IDJIError) {
                    ToastUtils.showToast("Takeoff Failed: ${error.description()}")
                }
            })
        }

        // Landing Button Click Listener
        btnLand?.setOnClickListener {
            basicAircraftControlVM.startLanding(object :
                CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(msg: EmptyMsg?) {
                    ToastUtils.showToast("Landing Succeeded")
                }

                override fun onFailure(error: IDJIError) {
                    ToastUtils.showToast("Landing Failed: ${error.description()}")
                }
            })
        }

        gimbalController = GimbalController(this)

        detectionOverlayView = findViewById(R.id.detection_overlay)

        Thread {
            detector.loadModel(applicationContext)
        }.start()

        primaryFpvWidget = findViewById(R.id.widget_primary_fpv)
        primaryFpvWidget.isCameraSourceNameVisible = false
        primaryFpvWidget.isCameraSourceSideVisible = false

        primaryFpvWidget.updateVideoSource(ComponentIndexType.LEFT_OR_MAIN)

        MediaDataCenter.getInstance().cameraStreamManager.addAvailableCameraUpdatedListener(
            availableCameraUpdatedListener
        )

        btnPitchUp = findViewById(R.id.gim_pitchup)
        btnPitchDown = findViewById(R.id.gim_pitchdown)
        btnYawLeft = findViewById(R.id.gim_yawleft)
        btnYawRight = findViewById(R.id.gim_yawright)

        switchToggleTracking = findViewById(R.id.enable_track)
        swtichToggleDetection = findViewById(R.id.enable_det)
        tvDetTime = findViewById(R.id.det_time_text)

        seekBarConfidence = findViewById(R.id.conf_slider)
        seekBarIou = findViewById(R.id.iou_slider)

        tvConfidence = findViewById(R.id.conf_slider_text)
        tvIou = findViewById(R.id.iou_slider_text)

        btnPidPitchKpIncrease = findViewById(R.id.pid_pitch_kp_incr)
        btnPidPitchKpDecrease = findViewById(R.id.pid_pitch_kp_decr)
        btnPidPitchKiIncrease = findViewById(R.id.pid_pitch_ki_incr)
        btnPidPitchKiDecrease = findViewById(R.id.pid_pitch_ki_decr)
        btnPidPitchKdIncrease = findViewById(R.id.pid_pitch_kd_incr)
        btnPidPitchKdDecrease = findViewById(R.id.pid_pitch_kd_decr)

        btnPidRollKpIncrease = findViewById(R.id.pid_roll_kp_incr)
        btnPidRollKpDecrease = findViewById(R.id.pid_roll_kp_decr)
        btnPidRollKiIncrease = findViewById(R.id.pid_roll_ki_incr)
        btnPidRollKiDecrease = findViewById(R.id.pid_roll_ki_decr)
        btnPidRollKdIncrease = findViewById(R.id.pid_roll_kd_incr)
        btnPidRollKdDecrease = findViewById(R.id.pid_roll_kd_decr)

        btnPidYawKpIncrease = findViewById(R.id.pid_yaw_kp_incr)
        btnPidYawKpDecrease = findViewById(R.id.pid_yaw_kp_decr)
        btnPidYawKiIncrease = findViewById(R.id.pid_yaw_ki_incr)
        btnPidYawKiDecrease = findViewById(R.id.pid_yaw_ki_decr)
        btnPidYawKdIncrease = findViewById(R.id.pid_yaw_kd_incr)
        btnPidYawKdDecrease = findViewById(R.id.pid_yaw_kd_decr)

        tvPidPitchKp = findViewById(R.id.pid_pitch_kp_text)
        tvPidPitchKi = findViewById(R.id.pid_pitch_ki_text)
        tvPidPitchKd = findViewById(R.id.pid_pitch_kd_text)

        tvPidRollKp = findViewById(R.id.pid_roll_kp_text)
        tvPidRollKi = findViewById(R.id.pid_roll_ki_text)
        tvPidRollKd = findViewById(R.id.pid_roll_kd_text)

        tvPidYawKp = findViewById(R.id.pid_yaw_kp_text)
        tvPidYawKi = findViewById(R.id.pid_yaw_ki_text)
        tvPidYawKd = findViewById(R.id.pid_yaw_kd_text)

        // Pitch
        setupPidTuneButton(
            btnPidPitchKpIncrease,
            tvPidPitchKp,
            "Kp",
            { gimbalController?.kpPitch ?: 100.0 },
            { gimbalController?.kpPitch = it },
            5.0
        )
        setupPidTuneButton(
            btnPidPitchKpDecrease,
            tvPidPitchKp,
            "Kp",
            { gimbalController?.kpPitch ?: 100.0 },
            { gimbalController?.kpPitch = it },
            -5.0
        )
        setupPidTuneButton(
            btnPidPitchKiIncrease,
            tvPidPitchKi,
            "Ki",
            { gimbalController?.kiPitch ?: 0.0 },
            { gimbalController?.kiPitch = it },
            0.1
        )
        setupPidTuneButton(
            btnPidPitchKiDecrease,
            tvPidPitchKi,
            "Ki",
            { gimbalController?.kiPitch ?: 0.0 },
            { gimbalController?.kiPitch = it },
            -0.1
        )
        setupPidTuneButton(
            btnPidPitchKdIncrease,
            tvPidPitchKd,
            "Kd",
            { gimbalController?.kdPitch ?: 0.0 },
            { gimbalController?.kdPitch = it },
            1.0
        )
        setupPidTuneButton(
            btnPidPitchKdDecrease,
            tvPidPitchKd,
            "Kd",
            { gimbalController?.kdPitch ?: 0.0 },
            { gimbalController?.kdPitch = it },
            -1.0
        )

        // Roll
        setupPidTuneButton(
            btnPidRollKpIncrease,
            tvPidRollKp,
            "Kp",
            { gimbalController?.kpRoll ?: 100.0 },
            { gimbalController?.kpRoll = it },
            5.0
        )
        setupPidTuneButton(
            btnPidRollKpDecrease,
            tvPidRollKp,
            "Kp",
            { gimbalController?.kpRoll ?: 100.0 },
            { gimbalController?.kpRoll = it },
            -5.0
        )
        setupPidTuneButton(
            btnPidRollKiIncrease,
            tvPidRollKi,
            "Ki",
            { gimbalController?.kiRoll ?: 0.0 },
            { gimbalController?.kiRoll = it },
            0.1
        )
        setupPidTuneButton(
            btnPidRollKiDecrease,
            tvPidRollKi,
            "Ki",
            { gimbalController?.kiRoll ?: 0.0 },
            { gimbalController?.kiRoll = it },
            -0.1
        )
        setupPidTuneButton(
            btnPidRollKdIncrease,
            tvPidRollKd,
            "Kd",
            { gimbalController?.kdRoll ?: 0.0 },
            { gimbalController?.kdRoll = it },
            1.0
        )
        setupPidTuneButton(
            btnPidRollKdDecrease,
            tvPidRollKd,
            "Kd",
            { gimbalController?.kdRoll ?: 0.0 },
            { gimbalController?.kdRoll = it },
            -1.0
        )

        // Yaw
        setupPidTuneButton(
            btnPidYawKpIncrease,
            tvPidYawKp,
            "Kp",
            { gimbalController?.kpYaw ?: 100.0 },
            { gimbalController?.kpYaw = it },
            5.0
        )
        setupPidTuneButton(
            btnPidYawKpDecrease,
            tvPidYawKp,
            "Kp",
            { gimbalController?.kpYaw ?: 100.0 },
            { gimbalController?.kpYaw = it },
            -5.0
        )
        setupPidTuneButton(
            btnPidYawKiIncrease,
            tvPidYawKi,
            "Ki",
            { gimbalController?.kiYaw ?: 0.0 },
            { gimbalController?.kiYaw = it },
            0.1
        )
        setupPidTuneButton(
            btnPidYawKiDecrease,
            tvPidYawKi,
            "Ki",
            { gimbalController?.kiYaw ?: 0.0 },
            { gimbalController?.kiYaw = it },
            -0.1
        )
        setupPidTuneButton(
            btnPidYawKdIncrease,
            tvPidYawKd,
            "Kd",
            { gimbalController?.kdYaw ?: 0.0 },
            { gimbalController?.kdYaw = it },
            1.0
        )
        setupPidTuneButton(
            btnPidYawKdDecrease,
            tvPidYawKd,
            "Kd",
            { gimbalController?.kdYaw ?: 0.0 },
            { gimbalController?.kdYaw = it },
            -1.0
        )

        setupGimbalButton(btnPitchUp, pitchSpeed = 15.0, yawSpeed = 0.0)
        setupGimbalButton(btnPitchDown, pitchSpeed = -15.0, yawSpeed = 0.0)
        setupGimbalButton(btnYawLeft, pitchSpeed = 0.0, yawSpeed = -15.0)
        setupGimbalButton(btnYawRight, pitchSpeed = 0.0, yawSpeed = 15.0)

        if (seekBarConfidence != null) {
            seekBarConfidence!!.setOnSeekBarChangeListener(object :
                SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(
                    seekBar: SeekBar?, progress: Int, fromUser: Boolean
                ) {
                    tvConfidence?.text = "Confidence: " + progress + "%"
                    detector.setConfidenceThreshold(progress / 100.0f)
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) {}

                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
            tvConfidence?.text = "Confidence: " + seekBarConfidence!!.getProgress() + "%"
            detector.setConfidenceThreshold(seekBarConfidence!!.progress / 100.0f)
        }

        if (seekBarIou != null) {
            seekBarIou!!.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(
                    seekBar: SeekBar?, progress: Int, fromUser: Boolean
                ) {
                    tvIou?.text = "IoU: " + progress + "%"
                    detector.setIouThreshold(progress / 100.0f)
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) {}

                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
            tvIou?.text = "IoU: " + seekBarIou!!.getProgress() + "%"
            detector.setIouThreshold(seekBarIou!!.progress / 100.0f)
        }

        setupToggle(swtichToggleDetection) { isChecked ->
            detectionActive = isChecked
            if (isChecked) {
                if (!detector.isModelLoaded()) {
                    Thread {
                        detector.loadModel(applicationContext)
                    }.start()
                }
                startFrameCapture()
                tvDetTime?.text = "Detection running: " + detTime + "ms"
            } else {
                stopFrameCapture()
                // tracker.reset()
                detectionOverlayView?.clear()
                tvDetTime?.text = "Detection not running"
            }
        }

        setupToggle(switchToggleTracking) { isChecked ->
            trackingActive = isChecked
            if (!isChecked) {
                // tracker.reset()
            }
        }
    }

    private val frameListener =
        ICameraStreamManager.CameraFrameListener { frameData, offset, length, width, height, format ->
            if (!detectionActive) return@CameraFrameListener

            val bitmap: Bitmap = createBitmap(width, height)
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(frameData, offset, length))

            val scaledBitmap: Bitmap = bitmap.scale(320, 320)
            bitmap.recycle()

            val result = detector.detect(scaledBitmap)
            scaledBitmap.recycle()

            val trackedTarget = if (trackingActive) {
                tracker.track(result.targets)
            } else {
                null
            }

            mainHandler.post {
                detTime = result.inferenceTimeMs
                tvDetTime?.text = "Detection running: " + detTime + "ms"
                
                if (trackingActive) {
                    if (trackedTarget != null) {
                        detectionOverlayView?.setTargets(listOf(trackedTarget))
                        gimbalController?.updateTarget(trackedTarget.rect, System.currentTimeMillis())
                        Log.d("Detection", "Tracker predicted target: ${trackedTarget.rect}")
                    } else {
                        detectionOverlayView?.clear()
                        gimbalController?.updateTarget(null, System.currentTimeMillis())
                        Log.d("Detection", "No target tracked")
                    }
                } else {
                    // When tracking is disabled, simply show raw detections
                    detectionOverlayView?.setTargets(result.targets)
                    gimbalController?.updateTarget(null, System.currentTimeMillis())
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
        button: Button?, pitchSpeed: Double, yawSpeed: Double
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

    fun rotateGimbalBySpeed(pitchSpeed: Double, yawSpeed: Double, rollSpeed: Double = 0.0) {
        val rotation = GimbalSpeedRotation(pitchSpeed, yawSpeed, rollSpeed, CtrlInfo())
        KeyManager.getInstance().performAction(
            KeyTools.createKey(GimbalKey.KeyRotateBySpeed, gimbal),
            rotation,
            object : CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(msg: EmptyMsg?) {}
                override fun onFailure(error: IDJIError) {}
            })
    }

    private fun setupPidTuneButton(
        button: Button?,
        textView: TextView?,
        label: String,
        getValue: () -> Double,
        setValue: (Double) -> Unit,
        delta: Double
    ) {
        updatePidText(textView, label, getValue())

        button?.setOnClickListener {
            val newValue = getValue() + delta
            setValue(newValue)
            updatePidText(textView, label, getValue())
        }
    }

    private fun updatePidText(textView: TextView?, label: String, value: Double) {
        val formatted = if (value % 1.0 == 0.0) {
            String.format(Locale.US, "%s: %.0f", label, value)
        } else {
            String.format(Locale.US, "%s: %.2f", label, value)
        }
        textView?.text = formatted
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
        MediaDataCenter.getInstance().cameraStreamManager.removeAvailableCameraUpdatedListener(
            availableCameraUpdatedListener
        )
    }
}