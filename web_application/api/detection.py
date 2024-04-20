import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
import supervision as sv
from ultralytics import YOLO
from torchvision import transforms

from DDAMFN.networks.DDAM import DDAMNet


CLASSIFICATION_MODEL_PATH = "api/weights/emotion_model_8.pth"
DETECTION_MODEL_PATH = "api/weights/yolov8l-face.pt"
softmax = torch.nn.Softmax(1)
FRAME_SIZE = 640
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2 
AGGREGATE_BY_LAST_N = 10
CLASS_TO_IDX = {
    "anger": 0,
    "contempt": 1,
    "disgust": 2,
    "fear": 3,
    "joy": 4,
    "sadness": 5,
    "wonder": 6,
    "neutral": 7,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASSIFICATION_TRANSFORMS=transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


def classify_emotions(frame, model):
    frame = CLASSIFICATION_TRANSFORMS(frame).to("cuda")
    emotions = model(frame[None, :])[0].detach().cpu()
    return softmax(emotions)


def detect_faces(frame, model, verbose=False):
    return model.predict(frame, conf=0.5, verbose=verbose)


def process(path, detection_model, classification_model):
    cls_model = DDAMNet(8, pretrained=False)
    checkpoint = torch.load(classification_model)
    cls_model.load_state_dict(checkpoint['model_state_dict'])
    cls_model.eval()
    cls_model = cls_model.to("cuda")

    dtct_model = YOLO(detection_model)
    dtct_model = dtct_model.to("cuda")

    core_path = Path(path)
    output_path = str(core_path.parent / f"{core_path.stem}_output{core_path.suffix}")

    cam = cv2.VideoCapture(path)
    tracker = sv.ByteTrack()
    if not cam.isOpened():
        raise Exception("Can not read input")
    INITIAL_RES = (int(cam.get(3)), int(cam.get(4)))
    FPS = cam.get(cv2.CAP_PROP_FPS)
    FPS_COUNT = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    DEVICE = "cuda"
    fourcc =  cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, FPS, INITIAL_RES)

    frame_count = 0
    track_history = {}

    while True:
        success, frame = cam.read()
        if not success:
            raise Exception("Error during reading the frames")
        
        frame_count += 1
        if frame_count == FPS_COUNT:
            break

        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

        faces = detect_faces(frame, model=dtct_model, verbose=False)[0]
        faces = sv.Detections.from_ultralytics(faces)
        faces = tracker.update_with_detections(faces)

        for (x1, y1, x2, y2), track_id in zip(faces.xyxy, faces.tracker_id):
            if track_id not in track_history:
                track_history[track_id] = torch.tensor([], dtype=torch.float32)
                
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            region_of_interest = Image.fromarray(rgb[y1:y1+h, x1:x1+w])
            emotions = classify_emotions(region_of_interest, model=cls_model)

            if track_history[track_id].shape[0] < AGGREGATE_BY_LAST_N:
                predicted_emotion = emotions.argmax().item()
                track_history[track_id] = torch.cat((track_history[track_id], emotions), dim=0)
            else:
                predicted_emotion = track_history[track_id].mean(dim=0).argmax().item()
                track_history[track_id] = track_history[track_id][1:]
                track_history[track_id] = torch.cat((track_history[track_id], emotions), dim=0)

            distribution = {idx: value for idx, value in enumerate(emotions.flatten())}
            distribution = {k: v.item() for k, v in sorted(distribution.items(), key=lambda x: x[1], reverse=True)}

            predicted_emotion = IDX_TO_CLASS[predicted_emotion]
            if predicted_emotion in ("anger", "contempt", "sadness"):
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color)
            cv2.putText(frame, predicted_emotion,
                        (x1, y1), FONT, FONT_SCALE, COLOR, THICKNESS)

        frame = cv2.resize(frame, INITIAL_RES)
        writer.write(frame)

    writer.release()
    cam.release()


parser = argparse.ArgumentParser(description='Emotion detection')
parser.add_argument("path_to_video", default="")
parser.add_argument("--detection_model", default=DETECTION_MODEL_PATH)
parser.add_argument("--classification_model", default=CLASSIFICATION_MODEL_PATH)
parser.add_argument("--online", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

if args.online is False:
    process(args.path_to_video, args.detection_model, args.classification_model)
else:
    pass

