import argparse
import datetime
import os
import time

import cv2
import torch
import torchvision
from torchvision.io import read_image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import functional as F

from pennfudan import PennFudanDataset


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def print_elapsed(start_time, s):
    elapsed = round(time.time() - start_time)
    rel_time = str(datetime.timedelta(seconds=elapsed))
    print(f"[{rel_time}]", s)


def train(device):
    start_time = time.time()
    num_classes = 2
    dataset = PennFudanDataset("datasets/PennFudanPed")

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 2

    for epoch in range(num_epochs):
        print_elapsed(start_time, f"Epoch {epoch + 1} started")
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print_elapsed(
            start_time,
            f"Epoch {epoch + 1} complete. Final batch loss: {losses.item():.4f}",
        )

    torch.save(model.state_dict(), "model_weights.pth")
    print_elapsed(start_time, "Model weights saved to 'model_weights.pth'")


def infer(device, path=None):
    num_classes = 2
    # model = get_model_instance_segmentation(num_classes)
    # model.load_state_dict(
    #     torch.load("model_weights.pth", weights_only=True, map_location=device)
    # )
    # model.to(device)
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model.eval()

    cap = cv2.VideoCapture(0)
    if path is None:
        cap = cv2.VideoCapture(0)
        image_files = None
    else:
        cap = None
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = sorted(
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(valid_extensions)
            ]
        )
    img_index = 0
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if img_index >= len(image_files):
                break
            frame = cv2.imread(image_files[img_index])
            img_index += 1
            if frame is None:
                continue

        start_time = time.perf_counter()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(device)

        with torch.no_grad():
            predictions = model([img_tensor])

        prediction = predictions[0]
        for i in range(len(prediction["boxes"])):
            score = prediction["scores"][i].item()
            if score > 0.8:
                box = prediction["boxes"][i].cpu().numpy().astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person: {score:.2f}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        end_time = time.perf_counter()
        ms_per_frame = (end_time - start_time) * 1000

        cv2.putText(
            frame,
            f"{ms_per_frame:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Webcam Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection PoC")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.train:
        train(device)
    elif args.infer:
        if not os.path.exists("model_weights.pth"):
            print(
                "Error: 'model_weights.pth' not found. Please run with --train first."
            )
        else:
            infer(device)  # , "datasets/PennFudanPed/PNGImages")
    else:
        print("Please provide a valid argument: --train or --infer")
