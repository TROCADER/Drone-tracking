package com.dji.sdk.sample.demo.gimbal;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * PyTorch TorchVision SSDLite & YOLO Human Detection Runner.
 * Performs inference on live video frames and generates red bounding box targets.
 */
public class PyTorchDetector {

    private static final String TAG = "PyTorchDetector";
    private static final String MODEL_ASSET_NAME = "ssdlite320_mobilenet_v3_xnnpack.pte";
    private static final int INPUT_SIZE = 320; // SSDLite / Detection standard resolution
    private float confidenceThreshold = 0.30f; // Confidence threshold
    private static final float IOU_THRESHOLD = 0.25f; // NMS IoU threshold

    public void setConfidenceThreshold(float threshold) {
        this.confidenceThreshold = threshold;
    }

    private static class NmsBox {
        RectF rect;
        float score;
        int classId;
        String label;

        NmsBox(RectF rect, float score, int classId, String label) {
            this.rect = rect;
            this.score = score;
            this.classId = classId;
            this.label = label;
        }
    }

    private Module module = null;
    private boolean isLoaded = false;

    public static class Recognition implements Comparable<Recognition> {
        public final int classIndex;
        public final String title;
        public final float confidence;

        public Recognition(int classIndex, String title, float confidence) {
            this.classIndex = classIndex;
            this.title = title;
            this.confidence = confidence;
        }

        @Override
        public int compareTo(Recognition o) {
            return Float.compare(o.confidence, this.confidence);
        }

        @Override
        public String toString() {
            return String.format(Locale.US, "%s (#%d): %.1f%%", title, classIndex, confidence * 100.0f);
        }
    }

    public static class DetectionResult {
        public final List<DetectionOverlayView.TargetBox> targets;
        public final List<Recognition> topRecognitions;
        public final long inferenceTimeMs;
        public final boolean success;
        public final String errorMessage;

        public DetectionResult(List<DetectionOverlayView.TargetBox> targets, List<Recognition> topRecognitions, long inferenceTimeMs) {
            this.targets = targets != null ? targets : Collections.emptyList();
            this.topRecognitions = topRecognitions != null ? topRecognitions : Collections.emptyList();
            this.inferenceTimeMs = inferenceTimeMs;
            this.success = true;
            this.errorMessage = null;
        }

        public DetectionResult(String errorMessage) {
            this.targets = Collections.emptyList();
            this.topRecognitions = Collections.emptyList();
            this.inferenceTimeMs = 0;
            this.success = false;
            this.errorMessage = errorMessage;
        }

        public Recognition getTopRecognition() {
            if (topRecognitions != null && !topRecognitions.isEmpty()) {
                return topRecognitions.get(0);
            }
            return null;
        }
    }

    /**
     * Initializes and loads the PyTorch model safely from assets.
     */
    public synchronized boolean loadModel(Context context) {
        if (isLoaded && module != null) {
            return true;
        }

        try {
            String modelPath = assetFilePath(context, MODEL_ASSET_NAME);
            Log.d(TAG, "Loading PyTorch model from: " + modelPath);

            module = Module.load(modelPath);
            Log.i(TAG, "Loaded PyTorch detection module successfully!");

            isLoaded = (module != null);
            return isLoaded;
        } catch (Throwable t) {
            Log.e(TAG, "Failed to load PyTorch model: " + t.getMessage(), t);
            isLoaded = false;
            module = null;
            return false;
        }
    }

    public boolean isModelLoaded() {
        return isLoaded && module != null;
    }

    /**
     * Runs human detection & inference on the provided video frame Bitmap.
     */
    public DetectionResult detect(Bitmap bitmap) {
        if (!isModelLoaded() || bitmap == null) {
            return new DetectionResult("Model is not loaded or frame is null");
        }

        Bitmap scaledBitmap = null;
        try {
            scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

            // The PyTorch model (ssdlite) trace expects an unbatched 3D tensor [3, H, W]
            // because it internally unsqueezes to add the batch dimension.
            final java.nio.FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(3 * INPUT_SIZE * INPUT_SIZE);
            bitmapToFloatBuffer(scaledBitmap, inTensorBuffer);
            
            final Tensor inputTensor = Tensor.fromBlob(
                    inTensorBuffer,
                    new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}
            );

            long startTime = System.currentTimeMillis();
            
            // Forward pass
            EValue[] outputs = module.forward(EValue.from(inputTensor));
            long elapsedTime = System.currentTimeMillis() - startTime;

            List<DetectionOverlayView.TargetBox> targetBoxes = new ArrayList<>();
            List<Recognition> topRecognitions = new ArrayList<>();
            List<NmsBox> rawBoxes = new ArrayList<>();

            if (outputs != null && outputs.length >= 2) {
                float[] boxesData = outputs[0].toTensor().getDataAsFloatArray();
                float[] scoresData = outputs[1].toTensor().getDataAsFloatArray();

                // According to python code, the output is decoded_boxes [3234, 4] and scores [3234, 91]
                int numBoxes = boxesData.length / 4;
                int numClasses = scoresData.length / numBoxes; // Should be 91
                
                // SSD outputs typically need to be decoded. If the model outputs
                // raw boxes, they usually need to be adjusted with priors/anchors.
                // Assuming the output is already decoded bounding boxes:

                for (int i = 0; i < numBoxes; i++) {
                    // Start from class 1 (skip background class 0)
                    float maxScore = -1.0f;
                    int maxClassId = -1;
                    
                    for (int c = 1; c < numClasses; c++) {
                        float score = scoresData[i * numClasses + c];
                        if (score > maxScore) {
                            maxScore = score;
                            maxClassId = c;
                        }
                    }
                    
                    // We specifically check if the highest scoring class is a person (index 1)
                    if (maxScore >= confidenceThreshold && maxClassId == 1) {
                        // Coordinates are 0..320, we need to normalize them to 0..1
                        float x1 = boxesData[i * 4] / INPUT_SIZE;
                        float y1 = boxesData[i * 4 + 1] / INPUT_SIZE;
                        float x2 = boxesData[i * 4 + 2] / INPUT_SIZE;
                        float y2 = boxesData[i * 4 + 3] / INPUT_SIZE;

                        RectF rect = new RectF(
                                Math.max(0.0f, Math.min(1.0f, x1)),
                                Math.max(0.0f, Math.min(1.0f, y1)),
                                Math.min(1.0f, Math.max(0.0f, x2)),
                                Math.min(1.0f, Math.max(0.0f, y2))
                        );
                        rawBoxes.add(new NmsBox(rect, maxScore, maxClassId, "Human Target"));
                    }
                }
            }

            // Apply NMS to collected boxes
            if (!rawBoxes.isEmpty()) {
                List<NmsBox> nmsFiltered = applyNMS(rawBoxes, IOU_THRESHOLD);
                for (NmsBox box : nmsFiltered) {
                    // One final check to make sure the boxes are somewhat sane sizes
                    float boxArea = (box.rect.right - box.rect.left) * (box.rect.bottom - box.rect.top);
                    if (boxArea > 0.001f && boxArea < 0.95f) {
                        targetBoxes.add(new DetectionOverlayView.TargetBox(box.rect, box.label, box.score));
                        topRecognitions.add(new Recognition(box.classId, box.label, box.score));
                        break; // Gives single target
                    }
                }
            }

            // Fallback for placeholder models
            if (targetBoxes.isEmpty() && !topRecognitions.isEmpty()) {
                Recognition top = topRecognitions.get(0);
                RectF fallbackRect = new RectF(0.25f, 0.20f, 0.75f, 0.80f);
                targetBoxes.add(new DetectionOverlayView.TargetBox(fallbackRect, "Human Target", Math.max(0.25f, top.confidence)));
            }

            Log.d(TAG, "Detection completed in " + elapsedTime + " ms, targets found: " + targetBoxes.size());
            return new DetectionResult(targetBoxes, topRecognitions, elapsedTime);

        } catch (Throwable t) {
            Log.e(TAG, "Error during PyTorch human detection: " + t.getMessage(), t);
            return new DetectionResult("Inference error: " + t.getMessage());
        } finally {
            if (scaledBitmap != null && scaledBitmap != bitmap) {
                scaledBitmap.recycle();
            }
        }
    }


    private void bitmapToFloatBuffer(Bitmap bitmap, java.nio.FloatBuffer buffer) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int area = width * height;
        int[] pixels = new int[area];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < area; i++) {
            int pixel = pixels[i];
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            // Planar CHW layout: R plane, then G plane, then B plane
            buffer.put(i, r);
            buffer.put(area + i, g);
            buffer.put(2 * area + i, b);
        }
    }

    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : logits) {
            if (val > max) {
                max = val;
            }
        }
        float sum = 0.0f;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        if (sum > 0) {
            for (int i = 0; i < exp.length; i++) {
                exp[i] /= sum;
            }
        }
        return exp;
    }

    /**
     * Copies asset file to internal device storage for PyTorch C++ loader access.
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        try (InputStream is = context.getAssets().open(assetName)) {
            long assetLength = is.available();
            if (file.exists() && file.length() == assetLength && assetLength > 0) {
                return file.getAbsolutePath();
            }
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[8 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
        }
        return file.getAbsolutePath();
    }

    private List<NmsBox> applyNMS(List<NmsBox> boxes, float iouThreshold) {
        Collections.sort(boxes, new java.util.Comparator<NmsBox>() {
            @Override
            public int compare(NmsBox b1, NmsBox b2) {
                return Float.compare(b2.score, b1.score);
            }
        });

        List<NmsBox> selected = new ArrayList<>();
        for (NmsBox box : boxes) {
            boolean keep = true;
            for (NmsBox selectedBox : selected) {
                if (calculateIoU(box.rect, selectedBox.rect) > iouThreshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.add(box);
            }
        }
        return selected;
    }

    private float calculateIoU(RectF a, RectF b) {
        float x1 = Math.max(a.left, b.left);
        float y1 = Math.max(a.top, b.top);
        float x2 = Math.min(a.right, b.right);
        float y2 = Math.min(a.bottom, b.bottom);

        float intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);

        if (areaA + areaB - intersectionArea <= 0) return 0.0f;
        return intersectionArea / (areaA + areaB - intersectionArea);
    }
}
