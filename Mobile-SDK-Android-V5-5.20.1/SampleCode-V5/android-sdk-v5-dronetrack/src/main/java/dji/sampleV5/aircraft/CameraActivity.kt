package dji.sampleV5.aircraft

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
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
import dji.v5.manager.interfaces.ICameraStreamManager.AvailableCameraUpdatedListener
import dji.v5.ux.core.widget.fpv.FPVWidget

class CameraActivity : AppCompatActivity() {

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

        primaryFpvWidget = findViewById(R.id.widget_primary_fpv)
        primaryFpvWidget.isCameraSourceNameVisible = false
        primaryFpvWidget.isCameraSourceSideVisible = false

        primaryFpvWidget.updateVideoSource(ComponentIndexType.LEFT_OR_MAIN)

        MediaDataCenter.getInstance().cameraStreamManager.addAvailableCameraUpdatedListener(availableCameraUpdatedListener)

        val btnPitchUp = findViewById<Button>(R.id.btn_pitchup)
        val btnPitchDown = findViewById<Button>(R.id.btn_pitchdown)
        val btnYawLeft = findViewById<Button>(R.id.btn_yawleft)
        val btnYawRight = findViewById<Button>(R.id.btn_yawright)

        setupGimbalButton(btnPitchUp, pitchSpeed = 15.0, yawSpeed = 0.0)
        setupGimbalButton(btnPitchDown, pitchSpeed = -15.0, yawSpeed = 0.0)
        setupGimbalButton(btnYawLeft, pitchSpeed = 0.0, yawSpeed = -15.0)
        setupGimbalButton(btnYawRight, pitchSpeed = 0.0, yawSpeed = 15.0)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupGimbalButton(
        button: Button,
        pitchSpeed: Double,
        yawSpeed: Double
    ) {
        button.setOnTouchListener { _, event ->
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

    override fun onDestroy() {
        super.onDestroy()
        MediaDataCenter.getInstance().cameraStreamManager.removeAvailableCameraUpdatedListener(availableCameraUpdatedListener)
    }
}