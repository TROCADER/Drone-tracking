package dji.sampleV5.aircraft

import dji.sdk.keyvalue.value.common.RectF
import kotlin.math.max
import kotlin.math.min

fun calculateIoU(a: RectF, b: RectF): Float {
    val aLeft = a.left ?: 0.0
    val aTop = a.top ?: 0.0
    val aRight = a.right ?: 0.0
    val aBottom = a.bottom ?: 0.0

    val bLeft = b.left ?: 0.0
    val bTop = b.top ?: 0.0
    val bRight = b.right ?: 0.0
    val bBottom = b.bottom ?: 0.0

    val x1 = max(aLeft, bLeft)
    val y1 = max(aTop, bTop)
    val x2 = min(aRight, bRight)
    val y2 = min(aBottom, bBottom)

    val intersectionArea = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    val areaA = max(0.0, aRight - aLeft) * max(0.0, aBottom - aTop)
    val areaB = max(0.0, bRight - bLeft) * max(0.0, bBottom - bTop)

    val unionArea = areaA + areaB - intersectionArea
    if (unionArea <= 0) return 0.0f
    return (intersectionArea / unionArea).toFloat()
}
