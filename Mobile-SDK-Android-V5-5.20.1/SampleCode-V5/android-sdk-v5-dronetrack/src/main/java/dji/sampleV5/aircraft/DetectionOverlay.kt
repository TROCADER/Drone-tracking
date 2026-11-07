package dji.sampleV5.aircraft

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import java.util.Locale
import kotlin.math.min

class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6.0f
    }

    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 10.0f
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34.0f
        isFakeBoldText = true
    }

    private val targets: MutableList<Detector.TargetBox> = ArrayList()

    fun setTargets(newTargets: List<Detector.TargetBox>?) {
        synchronized(targets) {
            targets.clear()
            if (newTargets != null) {
                targets.addAll(newTargets)
            }
        }
        postInvalidate()
    }

    fun clear() {
        synchronized(targets) {
            targets.clear()
        }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val viewWidth = width
        val viewHeight = height

        if (viewWidth <= 0 || viewHeight <= 0) {
            return
        }

        synchronized(targets) {
            for ((rect, label, score) in targets) {
                val l = rect.left ?: 0.0
                val t = rect.top ?: 0.0
                val r = rect.right ?: 0.0
                val b = rect.bottom ?: 0.0

                // Convert normalized coordinates (0.0 - 1.0) to canvas pixels
                val left = (l * viewWidth).toFloat().coerceIn(0f, viewWidth.toFloat())
                val top = (t * viewHeight).toFloat().coerceIn(0f, viewHeight.toFloat())
                val right = (r * viewWidth).toFloat().coerceIn(0f, viewWidth.toFloat())
                val bottom = (b * viewHeight).toFloat().coerceIn(0f, viewHeight.toFloat())

                if (right <= left || bottom <= top) {
                    continue
                }

                canvas.drawRect(left, top, right, bottom, boxPaint)

                val boxW = right - left
                val boxH = bottom - top
                val cornerLen = min(boxW, boxH) * 0.22f

                // Top left
                canvas.drawLine(left, top, left + cornerLen, top, cornerPaint)
                canvas.drawLine(left, top, left, top + cornerLen, cornerPaint)

                // Top right
                canvas.drawLine(right, top, right - cornerLen, top, cornerPaint)
                canvas.drawLine(right, top, right, top + cornerLen, cornerPaint)

                // Bottom left
                canvas.drawLine(left, bottom, left + cornerLen, bottom, cornerPaint)
                canvas.drawLine(left, bottom, left, bottom - cornerLen, cornerPaint)

                // Bottom right
                canvas.drawLine(right, bottom, right - cornerLen, bottom, cornerPaint)
                canvas.drawLine(right, bottom, right, bottom - cornerLen, cornerPaint)

                val labelText = String.format(
                    Locale.US, "%s %.0f%%",
                    label.uppercase(Locale.US),
                    score * 100.0f
                )

                val textWidth = textPaint.measureText(labelText)
                val textHeight = textPaint.textSize
                val padding = 12.0f

                val bgLeft = left
                var bgBottom = top
                var bgTop = top - textHeight - (padding * 2)
                if (bgTop < 0) {
                    bgTop = top
                    bgBottom = top + textHeight + (padding * 2)
                }
                val bgRight = bgLeft + textWidth + (padding * 2)

                canvas.drawRect(bgLeft, bgTop, bgRight, bgBottom, labelBgPaint)
                canvas.drawText(labelText, bgLeft + padding, bgBottom - padding, textPaint)
            }
        }
    }
}
