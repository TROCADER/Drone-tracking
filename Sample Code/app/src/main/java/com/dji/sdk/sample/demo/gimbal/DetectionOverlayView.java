package com.dji.sdk.sample.demo.gimbal;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Transparent Overlay View that draws red bounding boxes and target tracking reticles
 * around detected humans/targets on top of the live video feed.
 */
public class DetectionOverlayView extends View {

    public static class TargetBox {
        public final RectF rect; // Normalized coordinates (0.0 to 1.0)
        public final String label;
        public final float confidence;

        public TargetBox(RectF rect, String label, float confidence) {
            this.rect = rect;
            this.label = label;
            this.confidence = confidence;
        }
    }

    private Paint boxPaint;
    private Paint cornerPaint;
    private Paint labelBgPaint;
    private Paint textPaint;

    private final List<TargetBox> targets = new ArrayList<>();

    public DetectionOverlayView(Context context) {
        super(context);
        init();
    }

    public DetectionOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public DetectionOverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        // Red main bounding box paint
        boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxPaint.setColor(Color.RED);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6.0f);

        // Bright red thick corner accents
        cornerPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        cornerPaint.setColor(Color.RED);
        cornerPaint.setStyle(Paint.Style.STROKE);
        cornerPaint.setStrokeWidth(10.0f);

        // Filled red background for text label
        labelBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        labelBgPaint.setColor(Color.RED);
        labelBgPaint.setStyle(Paint.Style.FILL);

        // Text paint for label
        textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(34.0f);
        textPaint.setFakeBoldText(true);
    }

    public void setTargets(List<TargetBox> newTargets) {
        synchronized (targets) {
            targets.clear();
            if (newTargets != null) {
                targets.addAll(newTargets);
            }
        }
        postInvalidate();
    }

    public void clear() {
        synchronized (targets) {
            targets.clear();
        }
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        final int viewWidth = getWidth();
        final int viewHeight = getHeight();

        if (viewWidth <= 0 || viewHeight <= 0) {
            return;
        }

        synchronized (targets) {
            for (TargetBox target : targets) {
                if (target == null || target.rect == null) {
                    continue;
                }

                // Convert normalized coordinates (0.0 - 1.0) to canvas pixels
                float left = Math.max(0, target.rect.left * viewWidth);
                float top = Math.max(0, target.rect.top * viewHeight);
                float right = Math.min(viewWidth, target.rect.right * viewWidth);
                float bottom = Math.min(viewHeight, target.rect.bottom * viewHeight);

                if (right <= left || bottom <= top) {
                    continue;
                }

                // 1. Draw main red bounding box
                canvas.drawRect(left, top, right, bottom, boxPaint);

                // 2. Draw target lock corner accents (reticle styling)
                float boxW = right - left;
                float boxH = bottom - top;
                float cornerLen = Math.min(boxW, boxH) * 0.22f;

                // Top-Left corner
                canvas.drawLine(left, top, left + cornerLen, top, cornerPaint);
                canvas.drawLine(left, top, left, top + cornerLen, cornerPaint);

                // Top-Right corner
                canvas.drawLine(right, top, right - cornerLen, top, cornerPaint);
                canvas.drawLine(right, top, right, top + cornerLen, cornerPaint);

                // Bottom-Left corner
                canvas.drawLine(left, bottom, left + cornerLen, bottom, cornerPaint);
                canvas.drawLine(left, bottom, left, bottom - cornerLen, cornerPaint);

                // Bottom-Right corner
                canvas.drawLine(right, bottom, right - cornerLen, bottom, cornerPaint);
                canvas.drawLine(right, bottom, right, bottom - cornerLen, cornerPaint);

                // 3. Draw label text badge (e.g. "HUMAN 95%")
                String labelText = String.format(Locale.US, "%s %.0f%%",
                        target.label != null ? target.label.toUpperCase() : "HUMAN",
                        target.confidence * 100.0f);

                float textWidth = textPaint.measureText(labelText);
                float textHeight = textPaint.getTextSize();
                float padding = 12.0f;

                float bgLeft = left;
                float bgBottom = top;
                float bgTop = top - textHeight - (padding * 2);
                if (bgTop < 0) {
                    bgTop = top;
                    bgBottom = top + textHeight + (padding * 2);
                }
                float bgRight = bgLeft + textWidth + (padding * 2);

                // Draw background box for text
                canvas.drawRect(bgLeft, bgTop, bgRight, bgBottom, labelBgPaint);

                // Draw text inside label box
                canvas.drawText(labelText, bgLeft + padding, bgBottom - padding, textPaint);
            }
        }
    }
}
