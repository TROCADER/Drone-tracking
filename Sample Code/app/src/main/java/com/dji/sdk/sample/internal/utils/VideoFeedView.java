package com.dji.sdk.sample.internal.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.util.AttributeSet;
import android.view.TextureView;
import android.view.View;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import androidx.annotation.NonNull;
import dji.midware.usb.P3.UsbAccessoryService;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.thirdparty.rx.Observable;
import dji.thirdparty.rx.Subscription;
import dji.thirdparty.rx.android.schedulers.AndroidSchedulers;
import dji.thirdparty.rx.functions.Action1;

/**
 * VideoView will show the live video for the given video feed using TextureView.
 */
public class VideoFeedView extends TextureView implements TextureView.SurfaceTextureListener {
    //region Properties
    private final static String TAG = "VideoFeedView";
    private DJICodecManager codecManager = null;
    private VideoFeeder.VideoDataListener videoDataListener = null;
    private boolean isPrimaryVideoFeed = true;
    private View coverView;
    private final long WAIT_TIME = 500; // Half of a second
    private final AtomicLong lastReceivedFrameTime = new AtomicLong(0);
    private final Observable<Long> timer =
        Observable.timer(100, TimeUnit.MICROSECONDS).observeOn(AndroidSchedulers.mainThread()).repeat();
    private Subscription subscription;

    private int currentWidth = 0;
    private int currentHeight = 0;
    //endregion

    //region Life-Cycle
    public VideoFeedView(Context context) {
        this(context, null, 0);
    }

    public VideoFeedView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public VideoFeedView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init(context);
    }

    public void setCoverView(View view) {
        coverView = view;
    }

    private void init(final Context context) {
        // Avoid the rendering exception in the Android Studio Preview view.
        if (isInEditMode()) {
            return;
        }

        setSurfaceTextureListener(this);

        videoDataListener = new VideoFeeder.VideoDataListener() {
            @Override
            public void onReceive(byte[] videoBuffer, int size) {
                lastReceivedFrameTime.set(System.currentTimeMillis());

                if (codecManager != null) {
                    codecManager.sendDataToDecoder(videoBuffer,
                                                   size,
                                                   isPrimaryVideoFeed
                                                   ? UsbAccessoryService.VideoStreamSource.Camera.getIndex()
                                                   : UsbAccessoryService.VideoStreamSource.Fpv.getIndex());
                }
            }
        };

        subscription = timer.subscribe(new Action1() {
            @Override
            public void call(Object o) {
                final long now = System.currentTimeMillis();
                final long elapsed = now - lastReceivedFrameTime.get();
                if (coverView != null) {
                    if (elapsed > WAIT_TIME && !ModuleVerificationUtil.isMavic2Product()) {
                        if (coverView.getVisibility() == INVISIBLE) {
                            coverView.setVisibility(VISIBLE);
                        }
                    } else {
                        if (coverView.getVisibility() == VISIBLE) {
                            coverView.setVisibility(INVISIBLE);
                        }
                    }
                }
            }
        });
    }

    @Override
    public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }
        currentWidth = width;
        currentHeight = height;
        if (codecManager == null) {
            codecManager = new DJICodecManager(getContext().getApplicationContext(),
                    surface,
                    width,
                    height,
                    isPrimaryVideoFeed
                            ? UsbAccessoryService.VideoStreamSource.Camera
                            : UsbAccessoryService.VideoStreamSource.Fpv);
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }
        if (currentWidth != width || currentHeight != height) {
            currentWidth = width;
            currentHeight = height;
            if (codecManager != null) {
                codecManager.cleanSurface();
                codecManager.destroyCodec();
            }
            codecManager = new DJICodecManager(getContext().getApplicationContext(),
                    surface,
                    width,
                    height,
                    isPrimaryVideoFeed
                            ? UsbAccessoryService.VideoStreamSource.Camera
                            : UsbAccessoryService.VideoStreamSource.Fpv);
        }
    }

    @Override
    public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
        if (codecManager != null) {
            codecManager.cleanSurface();
            codecManager.destroyCodec();
            codecManager = null;
        }
        currentWidth = 0;
        currentHeight = 0;
        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {
    }

    public VideoFeeder.VideoDataListener registerLiveVideo(VideoFeeder.VideoFeed videoFeed, boolean isPrimary) {
        isPrimaryVideoFeed = isPrimary;

        if (videoDataListener != null && videoFeed != null && !videoFeed.getListeners().contains(videoDataListener)) {
            videoFeed.addVideoDataListener(videoDataListener);
            return videoDataListener;
        }
        return null;
    }

    public void changeSourceResetKeyFrame() {
        if (codecManager != null) {
            codecManager.resetKeyFrame();
        }
    }

    /**
     * Captures the current live frame bitmap from the video feed for AI inference.
     */
    public Bitmap captureFrame(int reqWidth, int reqHeight) {
        if (!isAvailable()) {
            return null;
        }
        try {
            Bitmap bmp = getBitmap(reqWidth, reqHeight);
            if (bmp == null) {
                bmp = getBitmap();
            }
            return bmp;
        } catch (Exception e) {
            try {
                return getBitmap();
            } catch (Exception ex) {
                return null;
            }
        }
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        if (subscription != null && !subscription.isUnsubscribed()) {
            subscription.unsubscribe();
        }
        if (codecManager != null) {
            codecManager.cleanSurface();
            codecManager.destroyCodec();
            codecManager = null;
        }
        VideoFeeder.getInstance().destroy();
    }
}
