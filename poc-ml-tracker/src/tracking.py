from point_tracker import Tracker, weighted_mean_shift
from time import perf_counter
import numpy as np
import pandas as pd
import configparser
from pathlib import Path
import cv2
import os
import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()


PERSON_CLASS = 1
assert weights.meta["categories"][PERSON_CLASS] == "person"


DATASET = Path("./datasets/MOT15/train/")
SEQ = "PETS09-S2L1"


def read_data(seq_path: Path):
    gt_path = seq_path / "gt" / "gt.txt"
    names = [
        "frame",
        "track_id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
    ]

    config = configparser.ConfigParser()
    config.read(seq_path / "seqinfo.ini")
    meta = dict(config["Sequence"])
    gt_data = pd.read_csv(gt_path, names=names, usecols=range(len(names)))

    return meta, gt_data


def predict_on_cv2_image(frame):
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_tensor = to_tensor(rgb_frame)

    with torch.no_grad():
        predictions = model([image_tensor])
    prediction = predictions[0]

    scale_tensor = torch.tensor(
        [width, height, width, height],
        dtype=torch.float32,
        device=prediction["boxes"].device,
    )
    prediction["boxes"] = prediction["boxes"] / scale_tensor
    return prediction


def compute_smoothed_centers(prediction, threshold, bandwidth):
    MAX_ITERS = 150
    boxes = prediction["boxes"]  # Bounding boxes: [N, 4]
    labels = prediction["labels"]  # Class IDs: [N]
    scores = prediction["scores"]  # Confidence scores: [N]

    mask = (scores >= threshold) & (labels == PERSON_CLASS)
    max_score_index = np.argmax(scores)
    x1, _, x2, _ = boxes[max_score_index]

    center_xs = 0.5 * (boxes[:, 0] + boxes[:, 2])
    center_ys = 0.5 * (boxes[:, 1] + boxes[:, 3])

    cluster_centers, combinded_weights, _ = weighted_mean_shift(
        np.column_stack([center_xs[mask], center_ys[mask]]),
        scores[mask],
        max_iter=MAX_ITERS,
        bandwidth=bandwidth,
    )

    return cluster_centers[combinded_weights >= threshold]


def select_boxes(raw_predictions, detection_threshold, iou_threshold):
    keep_mask = (raw_predictions["scores"] >= detection_threshold) & (
        raw_predictions["labels"] == PERSON_CLASS
    )

    indices = torchvision.ops.nms(
        raw_predictions["boxes"][keep_mask],
        raw_predictions["scores"][keep_mask],
        iou_threshold,
    )

    return {k: v[keep_mask][indices] for k, v in raw_predictions.items()}


def run_video(seq_path: Path, gt_data: pd.DataFrame):
    DETETECTION_THRESHOLD = 0.2

    image_files = [item for item in os.scandir(seq_path / "img1") if item.is_file()]
    image_files.sort(key=lambda p: p.name)
    tracker = Tracker(
        max_distance=1,
        state_uncertainty=200,
        measurement_noise=35,
        process_noise=0.1,
        max_age=2,
    )

    for i, img_path in enumerate(image_files, start=1):
        start = perf_counter()
        frame = cv2.imread(img_path.path)
        prediction = predict_on_cv2_image(frame)
        if frame is None:
            continue

        selected = select_boxes(prediction, DETETECTION_THRESHOLD, 0.05)
        # draw_predicitons(frame, prediction)
        draw_tracks(
            frame,
            tracker.update(
                torchvision.ops.box_convert(
                    selected["boxes"], in_fmt="xyxy", out_fmt="xywh"
                )
            ),
        )
        end = perf_counter()
        print(round(1 / (end - start), 3))
        # draw_gt_bb(gt_data, i, frame)
        cv2.imshow("Object Detection Stream", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def draw_tracks(
    frame: np.ndarray,
    active_tracks: dict[int, dict],
    vel_scale: float = 1.0,
    color_point: tuple[int, int, int] = (0, 255, 0),
    color_arrow: tuple[int, int, int] = (0, 0, 255),
):
    img_h, img_w = frame.shape[:2]

    for tid, track in active_tracks.items():

        x, y, w, h = track
        x1 = int(img_w * x)
        x2 = int(img_w * (x + w))
        y1 = int(img_h * y)
        y2 = int(img_h * (y + h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color_point)
        # cv2.arrowedLine(
        #     frame,
        #     start_pt,
        #     end_pt,
        #     color=color_arrow,
        #     thickness=2,
        #     tipLength=0.3,
        # )

        cv2.putText(
            frame,
            f"ID {tid}",
            (x1 + 8, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def overlay_kde(frame, kde, scale=0.25, alpha=0.5):
    h, w = frame.shape[:2]
    sw, sh = int(w * scale), int(h * scale)

    y, x = np.mgrid[0:sh, 0:sw]
    points = np.vstack([(x.ravel() / scale), (y.ravel() / scale)])

    density = kde(points).reshape(sh, sw)

    heat = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0, dst=frame)


def draw_predicitons(frame, prediction):
    boxes = prediction["boxes"]  # Bounding boxes: [N, 4]
    scores = prediction["scores"]  # Confidence scores: [N]

    # center_xs = 0.5 * (boxes[:, 0] + boxes[:, 2])
    # center_ys = 0.5 * (boxes[:, 1] + boxes[:, 3])

    # centers = np.vstack([center_xs[mask], center_ys[mask]])
    # kde = gaussian_kde(centers, weights=scores[mask])
    # overlay_kde(frame, kde, scale=1 / 8)
    img_h, img_w = frame.shape[:2]
    max_score = max(scores)
    for score, box in zip(scores, boxes):
        x1, y1, x2, y2 = box
        x1 = int(x1 * img_w)
        x2 = int(x2 * img_w)
        y1 = int(y1 * img_h)
        y2 = int(y2 * img_h)

        p = score / max_score

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, int(255 * p), 0), 1)
    #     # text = f"Class {label.item()}: {score.item():.2f}"
    # cv2.putText(
    #     frame,
    #     text,
    #     (x, max(y - 10, 0)),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    #     2,
    # )

    # cluster_centers, combinded_weights, _ = weighted_mean_shift(
    #     np.column_stack([center_xs[mask], center_ys[mask]]),
    #     scores[mask],
    #     max_iter=150,
    #     bandwidth=bandwidth,
    # )

    # for (cx, cy), combinded_weight in zip(cluster_centers, combinded_weights):
    #     cv2.circle(frame, (int(round(cx)), int(round(cy))), 3, (255, 0, 255), -1)
    #     text = str(round(combinded_weight, ndigits=3))
    #     cv2.putText(
    #         frame,
    #         text,
    #         (int(cx), max(int(cy) - 10, 0)),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         (0, 255, 0),
    #         2,
    #     )


def draw_gt_bb(gt_data, i, frame):
    for _, row in gt_data[gt_data["frame"] == i].iterrows():
        x1 = int(row["bb_left"])
        y1 = int(row["bb_top"])

        x2 = int(x1 + row["bb_width"])
        y2 = int(y1 + row["bb_height"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        cv2.putText(
            frame,
            str(int(row["track_id"])),
            (x1, y1),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            0,
        )


meta, gt_data = read_data(DATASET / SEQ)
run_video(DATASET / SEQ, gt_data)
