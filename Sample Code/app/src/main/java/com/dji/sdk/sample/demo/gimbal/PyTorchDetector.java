package com.dji.sdk.sample.demo.gimbal;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

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
    private static final String MODEL_ASSET_NAME = "model_pytorch.pt";
    private static final int INPUT_SIZE = 320; // SSDLite / Detection standard resolution
    private static final float CONFIDENCE_THRESHOLD = 0.25f; // Confidence threshold

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
            TensorImageUtils.bitmapToFloatBuffer(
                    scaledBitmap,
                    0, 0, INPUT_SIZE, INPUT_SIZE,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    inTensorBuffer,
                    0
            );
            final Tensor inputTensor = Tensor.fromBlob(
                    inTensorBuffer,
                    new long[]{3, INPUT_SIZE, INPUT_SIZE}
            );

            long startTime = System.currentTimeMillis();
            IValue outputValue;
            try {
                // Standard PyTorch forward pass (single tensor argument)
                outputValue = module.forward(IValue.from(inputTensor));
            } catch (Throwable t1) {
                try {
                    // Try Tuple/List of Tensors (often needed for some SSDLite wrapped inputs)
                    outputValue = module.forward(IValue.tupleFrom(IValue.from(inputTensor)));
                } catch (Throwable t2) {
                    // Fallback to IValue list
                    outputValue = module.forward(IValue.listFrom(inputTensor));
                }
            }
            long elapsedTime = System.currentTimeMillis() - startTime;

            List<DetectionOverlayView.TargetBox> targetBoxes = new ArrayList<>();
            List<Recognition> topRecognitions = new ArrayList<>();

            if (outputValue != null) {
                if (outputValue.isDictStringKey()) {
                    // SSDLite Dict output {"boxes": tensor, "scores": tensor, "labels": tensor}
                    Map<String, IValue> map = outputValue.toDictStringKey();
                    IValue boxesVal = map.get("boxes");
                    IValue scoresVal = map.get("scores");
                    IValue labelsVal = map.get("labels");

                    if (boxesVal != null && scoresVal != null && labelsVal != null) {
                        float[] boxesData = boxesVal.toTensor().getDataAsFloatArray();
                        float[] scoresData = scoresVal.toTensor().getDataAsFloatArray();
                        long[] labelsData = labelsVal.toTensor().getDataAsLongArray();

                        int count = scoresData.length;
                        for (int i = 0; i < count; i++) {
                            float score = scoresData[i];
                            if (score >= CONFIDENCE_THRESHOLD) {
                                int classId = (int) labelsData[i];
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
                                String label = (classId == 1) ? "Human Target" : getLabelName(classId);
                                targetBoxes.add(new DetectionOverlayView.TargetBox(rect, label, score));
                                topRecognitions.add(new Recognition(classId, label, score));
                            }
                        }
                    }
                } else if (outputValue.isTuple()) {
                    // SSDLite Tuple output (boxes, scores, labels)
                    IValue[] tuple = outputValue.toTuple();
                    if (tuple != null && tuple.length >= 3) {
                        float[] boxesData = tuple[0].toTensor().getDataAsFloatArray();
                        float[] scoresData = tuple[1].toTensor().getDataAsFloatArray();
                        long[] labelsData = tuple[2].toTensor().getDataAsLongArray();

                        int count = scoresData.length;
                        for (int i = 0; i < count; i++) {
                            float score = scoresData[i];
                            if (score >= CONFIDENCE_THRESHOLD) {
                                int classId = (int) labelsData[i];
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
                                String label = (classId == 1) ? "Human Target" : getLabelName(classId);
                                targetBoxes.add(new DetectionOverlayView.TargetBox(rect, label, score));
                                topRecognitions.add(new Recognition(classId, label, score));
                            }
                        }
                    }
                } else if (outputValue.isTensor()) {
                    final Tensor outputTensor = outputValue.toTensor();
                    final float[] scores = outputTensor.getDataAsFloatArray();
                    final long[] shape = outputTensor.shape();

                    if (scores != null && scores.length > 0) {
                        if (shape.length >= 2 && shape[shape.length - 1] >= 6) {
                            // Standard detection tensor shape [N, 6] (x1, y1, x2, y2, score, class_id)
                            int numDetections = (int) shape[shape.length - 2];
                            int numCols = (int) shape[shape.length - 1];

                            for (int i = 0; i < numDetections; i++) {
                                int offset = i * numCols;
                                float x1 = scores[offset];
                                float y1 = scores[offset + 1];
                                float x2 = scores[offset + 2];
                                float y2 = scores[offset + 3];
                                float score = scores[offset + 4];
                                int classId = (int) scores[offset + 5];

                                if (score >= CONFIDENCE_THRESHOLD) {
                                    if (x2 > 1.0f || y2 > 1.0f) {
                                        x1 /= INPUT_SIZE;
                                        y1 /= INPUT_SIZE;
                                        x2 /= INPUT_SIZE;
                                        y2 /= INPUT_SIZE;
                                    }
                                    RectF rect = new RectF(
                                            Math.max(0.0f, x1),
                                            Math.max(0.0f, y1),
                                            Math.min(1.0f, x2),
                                            Math.min(1.0f, y2)
                                    );
                                    String label = (classId == 1 || classId == 0) ? "Human Target" : getLabelName(classId);
                                    targetBoxes.add(new DetectionOverlayView.TargetBox(rect, label, score));
                                    topRecognitions.add(new Recognition(classId, label, score));
                                }
                            }
                        } else if (shape.length == 3 && shape[1] < shape[2] && shape[1] >= 5) {
                            // YOLO detection tensor shape [1, 84, 8400]
                            int numAnchors = (int) shape[2];

                            for (int anchor = 0; anchor < numAnchors; anchor++) {
                                // Person class score at index 4 (0:cx, 1:cy, 2:w, 3:h, 4:person_score)
                                float personScore = scores[4 * numAnchors + anchor];
                                if (personScore >= CONFIDENCE_THRESHOLD) {
                                    float cx = scores[anchor];
                                    float cy = scores[numAnchors + anchor];
                                    float w = scores[2 * numAnchors + anchor];
                                    float h = scores[3 * numAnchors + anchor];

                                    if (cx > 1.0f || w > 1.0f) {
                                        cx /= INPUT_SIZE;
                                        cy /= INPUT_SIZE;
                                        w /= INPUT_SIZE;
                                        h /= INPUT_SIZE;
                                    }

                                    float x1 = Math.max(0.0f, cx - w / 2.0f);
                                    float y1 = Math.max(0.0f, cy - h / 2.0f);
                                    float x2 = Math.min(1.0f, cx + w / 2.0f);
                                    float y2 = Math.min(1.0f, cy + h / 2.0f);

                                    RectF rect = new RectF(x1, y1, x2, y2);
                                    targetBoxes.add(new DetectionOverlayView.TargetBox(rect, "Human Target", personScore));
                                    topRecognitions.add(new Recognition(0, "Human Target", personScore));
                                }
                            }
                        } else {
                            // Classification logits
                            float[] probabilities = softmax(scores);
                            for (int i = 0; i < probabilities.length; i++) {
                                topRecognitions.add(new Recognition(i, "Human Target", probabilities[i]));
                            }
                            Collections.sort(topRecognitions);
                            Recognition top = topRecognitions.get(0);
                            RectF targetRect = new RectF(0.25f, 0.20f, 0.75f, 0.80f);
                            targetBoxes.add(new DetectionOverlayView.TargetBox(targetRect, "Human Target", Math.max(0.25f, top.confidence)));
                        }
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

    private String getLabelName(int classId) {
        if (classId == 1 || classId == 0) return "Human Target";
        return "Target #" + classId;
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
}
