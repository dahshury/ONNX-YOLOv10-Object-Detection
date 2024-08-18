import cv2

from yolov10 import YOLOv10

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "models/yolov10m.onnx"
yolov10_detector = YOLOv10(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov10_detector(frame)

    combined_img = yolov10_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
