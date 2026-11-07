package dji.sampleV5.aircraft

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import dji.sdk.keyvalue.value.common.ComponentIndexType
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
    }

    override fun onDestroy() {
        super.onDestroy()
        MediaDataCenter.getInstance().cameraStreamManager.removeAvailableCameraUpdatedListener(availableCameraUpdatedListener)
    }
}