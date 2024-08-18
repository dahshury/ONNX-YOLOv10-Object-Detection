import time
import cv2
import os
import yaml
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime

from yolov10.utils import xywh2xyxy, draw_detections, multiclass_nms


class YOLOv10:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = output[0][0]
        
        # Filter predictions with low confidence scores
        scores = predictions[:,4]
        predictions = predictions[scores > self.conf_threshold]
        

        if len(predictions) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = predictions[:, 5]

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def load_ground_truth_labels(self, label_path):
        if not os.path.exists(label_path):
            return np.array([]), np.array([])  # No labels exist
        labels = np.loadtxt(label_path).reshape(-1, 5)
        class_ids = labels[:, 0].astype(int)
        boxes = labels[:, 1:]
        return boxes, class_ids
    
    def yolo_to_xyxy(self, box, img_width, img_height):
        box=np.copy(box)
        # Scale normalized coordinates to image dimensions
        box[..., 0] *= img_width
        box[..., 1] *= img_height
        box[..., 2] *= img_width
        box[..., 3] *= img_height

        # Convert to xyxy format
        box[..., 0] -= box[..., 2] / 2  # x_min
        box[..., 1] -= box[..., 3] / 2  # y_min
        box[..., 2] += box[..., 0]  # x_max
        box[..., 3] += box[..., 1]  # y_max

        return box
    
    def calculate_iou(self, box1, box2, img_height, img_width):
        box2 = self.yolo_to_xyxy(box2, img_width, img_height)
        xA = max(box1[0], box2[0]) if max(box1[0], box2[0]) > 0 else 0
        yA = max(box1[1], box2[1]) if max(box1[1], box2[1]) > 0 else 0
        xB = min(box1[2], box2[2]) if min(box1[2], box2[2]) > 0 else 0
        yB = min(box1[3], box2[3]) if min(box1[3], box2[3]) > 0 else 0

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def evaluate_directory(self, dataset_dir, split, config_path, save=False, debug=False):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            num_classes = config.get('nc', None)
            
        if num_classes is None:
            raise ValueError("Number of classes (nc) is not defined in the configuration file.")

        thresholds = np.arange(0.5, 0.95, 0.05)
        aps = {thres: {cls: [] for cls in range(num_classes)} for thres in thresholds}
        
        images_path = os.path.join(dataset_dir, 'images', split)
        labels_path = images_path.replace('images', 'labels')
        
        all_iou_scores = []
        all_true_classes = []
        all_pred_classes = []
        all_pred_scores = []
        
        for file in os.listdir(images_path):
            if file.lower().endswith(('.png', '.webp', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(images_path, file)
                label_path = os.path.join(labels_path, file.replace(os.path.splitext(file)[1], '.txt'))
                img = cv2.imread(image_path)
                img_height, img_width = img.shape[:2]
                det_boxes, det_scores, det_classes = self.detect_objects(img)  # boxes are in xyxy
                gt_boxes, gt_class_ids = self.load_ground_truth_labels(label_path)
                
                if len(gt_boxes) > 0:
                    for gt_box, gt_class in zip(gt_boxes, gt_class_ids):
                        if len(det_boxes) > 0:
                            iou_scores = np.array([self.calculate_iou(det_box, gt_box, img_height, img_width) for det_box in det_boxes])
                            best_iou_idx = np.argmax(iou_scores)
                            best_iou = iou_scores[best_iou_idx]
                            
                            all_iou_scores.append(best_iou)
                            all_true_classes.append(gt_class)
                            all_pred_classes.append(det_classes[best_iou_idx])
                            all_pred_scores.append(det_scores[best_iou_idx])
                        else:
                            all_iou_scores.append(0)
                            all_true_classes.append(gt_class)
                            all_pred_classes.append(num_classes)  # Use num_classes as the "no detection" class
                            all_pred_scores.append(0)
                
                for det_box, det_score, det_class in zip(det_boxes, det_scores, det_classes):
                    if len(gt_boxes) == 0 or np.max([self.calculate_iou(det_box, gt_box, img_height, img_width) for gt_box in gt_boxes]) < 0.5:
                        all_iou_scores.append(0)
                        all_true_classes.append(num_classes)  # Use num_classes as the "no ground truth" class
                        all_pred_classes.append(det_class)
                        all_pred_scores.append(det_score)

        # Convert lists to numpy arrays for faster processing
        all_iou_scores, all_true_classes, all_pred_classes, all_pred_scores = [np.array(x) for x in [all_iou_scores, all_true_classes, all_pred_classes, all_pred_scores]]

        # Calculate metrics for each class
        mean_ap_50 = defaultdict(float)
        mean_ap_50_95 = defaultdict(float)
        precisions_curve = {}
        recalls_curve = {}
        precisions = {}
        recalls = {}
        
        for cls in range(num_classes):
            true_binary = (all_true_classes == cls).astype(int)
            pred_scores_cls = np.where(all_pred_classes == cls, all_pred_scores, 0)

            if np.sum(true_binary) > 0:  # Only calculate if there are positive samples
                precision, recall, _ = precision_recall_curve(true_binary, pred_scores_cls)
                precisions_curve[cls] = precision
                recalls_curve[cls] = recall
                
                ap_per_iou = []
                for thres in thresholds:
                    # Sort predictions by score
                    sort_idx = np.argsort(-pred_scores_cls)
                    sorted_scores = pred_scores_cls[sort_idx]
                    sorted_true_binary = true_binary[sort_idx]
                    sorted_iou_scores = all_iou_scores[sort_idx]
                    
                    # Calculate precision and recall for all thresholds
                    tp = np.zeros_like(sorted_scores)
                    fp = np.zeros_like(sorted_scores)
                    for i, (score, true_class, iou) in enumerate(zip(sorted_scores, sorted_true_binary, sorted_iou_scores)):
                        if score > 0:  # Consider only non-zero predictions
                            if true_class == 1 and iou >= thres:
                                tp[i] = 1
                            else:
                                fp[i] = 1

                    tp_cumsum = np.cumsum(tp)
                    fp_cumsum = np.cumsum(fp)
                    recalls_iou = tp_cumsum / np.sum(true_binary)
                    precisions_iou = tp_cumsum / (tp_cumsum + fp_cumsum)

                    # Calculate AP
                    ap = np.trapz(precisions_iou, recalls_iou)
                    ap_per_iou.append(ap)

                    if thres == 0.5:
                        mean_ap_50[cls] = ap
                        precisions[cls] = precisions_iou
                        recalls[cls] = recalls_iou

                mean_ap_50_95[cls] = np.mean(ap_per_iou)
            else:
                precisions[cls] = []
                recalls[cls] = []
                mean_ap_50[cls] = 0
                mean_ap_50_95[cls] = 0

        # Create a single figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # 1. Plot confusion matrix
        cm = confusion_matrix(all_true_classes, all_pred_classes, labels=range(num_classes + 1))
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title('Confusion Matrix')
        fig.colorbar(im, ax=ax1)
        tick_marks = np.arange(num_classes + 1)
        ax1.set_xticks(tick_marks)
        ax1.set_xticklabels(list(range(num_classes)) + ['No Det'])
        ax1.set_yticks(tick_marks)
        ax1.set_yticklabels(list(range(num_classes)) + ['No GT'])
        ax1.set_xlabel('Predicted label')
        ax1.set_ylabel('True label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        # 2. Plot Precision-Recall Curve
        for cls in range(num_classes):
            if len(recalls_curve[cls]) > 0 and len(precisions_curve[cls]) > 0:
                ax2.plot(recalls_curve[cls], precisions_curve[cls], label=f'Class {cls}')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        
        plt.tight_layout()
        if save:
            plt.savefig('evaluation_results.png')
        plt.show()
        plt.close()
        
        # Calculate and print mean average precision
        map_50 = np.mean(list(mean_ap_50.values()))
        map_50_95 = np.mean(list(mean_ap_50_95.values()))
        
        print(f"Mean Average Precision @0.5: {map_50:.4f}")
        print(f"Mean Average Precision @0.5:0.95: {map_50_95:.4f}")

if __name__ == '__main__':
    from imread_from_url import imread_from_url

    model_path = "../models/YOLOv10m.onnx"

    # Initialize YOLOv10 object detector
    YOLOv10_detector = YOLOv10(model_path, conf_thres=0.3, iou_thres=0.5)

    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)

    # Detect Objects
    YOLOv10_detector(img)

    # Draw detections
    combined_img = YOLOv10_detector.draw_detections(img)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
