package dji.sampleV5.aircraft

import dji.sampleV5.aircraft.Detector.TargetBox
import dji.sdk.keyvalue.value.common.RectF
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get

class Tracker {
    private var kFilter: Kalman = Kalman(RectF(0.4,0.4,0.6,0.6))

    fun track(targets: List<TargetBox>): RectF {
        kFilter.predict()


        val trackedRect = kFilter.currentEstimateRect()

        // TODO: IoU == 0 for all boxes needs to be considered
        val selectedBox = targets.maxByOrNull { calculateIoU(it.rect, trackedRect) }
        if (selectedBox != null) {
            kFilter.update(
                mk.ndarray(
                    mk[
                        selectedBox.rect.top,
                        selectedBox.rect.left,
                        selectedBox.rect.right - selectedBox.rect.left,
                        selectedBox.rect.bottom - selectedBox.rect.top
                    ]
                )
            )
        }

        return kFilter.currentEstimateRect()
    }


    //TODO: figure reset out
}