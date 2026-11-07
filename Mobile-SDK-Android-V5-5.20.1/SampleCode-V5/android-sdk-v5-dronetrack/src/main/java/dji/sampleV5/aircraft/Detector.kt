package dji.sampleV5.aircraft

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import dji.sdk.keyvalue.value.common.RectF
import org.pytorch.executorch.EValue
import org.pytorch.executorch.EValue.Companion.from
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor.Companion.allocateFloatBuffer
import org.pytorch.executorch.Tensor.Companion.fromBlob
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import androidx.core.graphics.scale

class Detector {

    private val TAG = "Detector"
    private val MODEL_ASSET_NAME = "ssdlite320_mobilenet_v3_xnnpack.pte"
    private val INPUT_SIZE = 320 // SSDLite resolution 320x320
    private var confidenceThreshold = 0.30f
    private var iouThreshold = 0.25f

    private var isLoaded = false
    private var module: Module? = null

    fun setConfidenceThreshold(threshold: Float) {
        this.confidenceThreshold = threshold
    }

    fun setIouThreshold(threshold: Float) {
        this.iouThreshold = threshold
    }

    class NmsBox(val rect: RectF, val score: Float, val classId: Int, val label: String)

    data class TargetBox(val rect: RectF, val label: String, val score: Float)

    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (!file.exists() || file.length() == 0L) {
            context.assets.open(assetName).use { input ->
                file.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    @Synchronized
    fun loadModel(context: Context): Boolean {
        if (isLoaded && module != null) return true

        return try {
            val modelPath = assetFilePath(context, MODEL_ASSET_NAME)
            Log.d(TAG, "Loading ExecuTorch model from: $modelPath")
            module = Module.load(modelPath)
            isLoaded = true
            Log.i(TAG, "Loaded ExecuTorch detection module successfully!")
            true
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to load ExecuTorch model", t)
            isLoaded = false
            module = null
            false
        }
    }

    class Recognition(val classIndex: Int, val title: String, val confidence: Float) :
        Comparable<Recognition> {
        override fun compareTo(other: Recognition): Int {
            return other.confidence.compareTo(this.confidence)
        }

        override fun toString(): String {
            return String.format(
                Locale.US,
                "%s (#%d): %.1f%%",
                title,
                classIndex,
                confidence * 100.0f
            )
        }
    }

    class DetectionResult {
        val targets: List<TargetBox>
        val topRecognitions: List<Recognition>
        val inferenceTimeMs: Long
        val success: Boolean
        val errorMessage: String?

        constructor(
            targets: List<TargetBox>?,
            topRecognitions: List<Recognition>?,
            inferenceTimeMs: Long
        ) {
            this.targets = targets ?: emptyList()
            this.topRecognitions = topRecognitions ?: emptyList()
            this.inferenceTimeMs = inferenceTimeMs
            this.success = true
            this.errorMessage = null
        }

        constructor(errorMessage: String?) {
            this.targets = emptyList()
            this.topRecognitions = emptyList()
            this.inferenceTimeMs = 0
            this.success = false
            this.errorMessage = errorMessage
        }

        val topRecognition: Recognition?
            get() = topRecognitions.firstOrNull()
    }

    fun isModelLoaded(): Boolean {
        return isLoaded && module != null
    }

    fun detect(bitmap: Bitmap?): DetectionResult {
        if (!isModelLoaded() || bitmap == null) {
            return DetectionResult("Model is not loaded or frame is null")
        }

        var scaledBitmap: Bitmap? = null
        try {
            scaledBitmap = bitmap.scale(INPUT_SIZE, INPUT_SIZE)

//            Three color channels
            val inTensorBuffer = allocateFloatBuffer(3 * INPUT_SIZE * INPUT_SIZE)
            bitmapToFloatBuffer(scaledBitmap, inTensorBuffer)

            val inputTensor = fromBlob(
                inTensorBuffer,
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            )

            val startTime = System.currentTimeMillis()

            val outputs: Array<EValue> = module?.forward(from(inputTensor))
                ?: return DetectionResult("Forward pass returned null output")
            val elapsedTime = System.currentTimeMillis() - startTime

            val targetBoxes = mutableListOf<TargetBox>()
            val topRecognitions = mutableListOf<Recognition>()
            val rawBoxes = mutableListOf<NmsBox>()

            if (outputs.size >= 2) {
                val boxesData = outputs[0].toTensor().dataAsFloatArray
                val scoresData = outputs[1].toTensor().dataAsFloatArray

                val numBoxes = boxesData.size / 4
                val numClasses = if (numBoxes > 0) scoresData.size / numBoxes else 0

                for (i in 0 until numBoxes) {
                    var maxScore = -1.0f
                    var maxClassId = -1

                    for (c in 1 until numClasses) {
                        val score = scoresData[i * numClasses + c]
                        if (score > maxScore) {
                            maxScore = score
                            maxClassId = c
                        }
                    }

                    if (maxScore >= confidenceThreshold && maxClassId == 1) {
                        val x1 = (boxesData[i * 4] / INPUT_SIZE).toDouble().coerceIn(0.0, 1.0)
                        val y1 = (boxesData[i * 4 + 1] / INPUT_SIZE).toDouble().coerceIn(0.0, 1.0)
                        val x2 = (boxesData[i * 4 + 2] / INPUT_SIZE).toDouble().coerceIn(0.0, 1.0)
                        val y2 = (boxesData[i * 4 + 3] / INPUT_SIZE).toDouble().coerceIn(0.0, 1.0)

                        val rect = RectF(
                            min(x1, x2),
                            min(y1, y2),
                            max(x1, x2),
                            max(y1, y2)
                        )
                        rawBoxes.add(NmsBox(rect, maxScore, maxClassId, "Target"))
                    }
                }
            }

            if (rawBoxes.isNotEmpty()) {
                val nmsFiltered = applyNMS(rawBoxes, iouThreshold)
                for (box in nmsFiltered) {
                    val rect = box.rect
                    val left = rect.left ?: 0.0
                    val right = rect.right ?: 0.0
                    val top = rect.top ?: 0.0
                    val bottom = rect.bottom ?: 0.0

                    val boxArea = (right - left) * (bottom - top)
                    if (boxArea in 0.001..0.95) {
                        targetBoxes.add(TargetBox(box.rect, box.label, box.score))
                        topRecognitions.add(Recognition(box.classId, box.label, box.score))
                        break // Gives single target, remove to enable multi-box
                    }
                }
            }

            // Fallback for placeholder models
            if (targetBoxes.isEmpty() && topRecognitions.isNotEmpty()) {
                val top = topRecognitions[0]
                val fallbackRect = RectF(0.25, 0.20, 0.75, 0.80)
                targetBoxes.add(
                    TargetBox(
                        fallbackRect,
                        "Target",
                        max(0.25f, top.confidence)
                    )
                )
            }

//            Should be 1 as long as cap is enabled
            Log.d(
                TAG,
                "Detection completed in $elapsedTime ms, targets found: ${targetBoxes.size}"
            )
            return DetectionResult(targetBoxes, topRecognitions, elapsedTime)
        } catch (t: Throwable) {
            Log.e(TAG, "Error during detection: ${t.message}", t)
            return DetectionResult("Inference error: ${t.message}")
        } finally {
            if (scaledBitmap != null && scaledBitmap != bitmap) {
                scaledBitmap.recycle()
            }
        }
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap, buffer: FloatBuffer) {
        val width = bitmap.width
        val height = bitmap.height
        val area = width * height
        val pixels = IntArray(area)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in 0 until area) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            // CHW layout: R -> G -> B
            buffer.put(i, r)
            buffer.put(area + i, g)
            buffer.put(2 * area + i, b)
        }
    }

    private fun applyNMS(boxes: MutableList<NmsBox>, iouThreshold: Float): MutableList<NmsBox> {
        boxes.sortByDescending { it.score }

        val selected = mutableListOf<NmsBox>()
        for (box in boxes) {
            var keep = true
            for (selectedBox in selected) {
                if (calculateIoU(box.rect, selectedBox.rect) > iouThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) {
                selected.add(box)
            }
        }
        return selected
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
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
}