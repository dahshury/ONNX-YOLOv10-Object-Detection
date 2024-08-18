![! ONNX YOLOv10 Object Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/raw/main/doc/img/detected_objects.jpg)
*Original image: [https://www.flickr.com/photos/nicolelee/19041780](https://www.flickr.com/photos/nicolelee/19041780)*

# Important
- The input images are directly resized to match the input size of the model. I skipped adding the pad to the input image, it might affect the accuracy of the model if the input image has a different aspect ratio compared to the input size of the model. Always try to get an input size with a ratio close to the input images you will use.

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```shell
git clone https://github.com/dahshury/ONNX-YOLOv10-Object-Detection
cd ONNX-YOLOv10-Object-Detection
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
Use the Google Colab notebook to convert the model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-yZg6hFg27uCPSycRCRtyezHhq_VAHxQ?usp=sharing)

You can convert the model using the following code after installing ultralitics (`pip install ultralytics`):
```python
from ultralytics import YOLO

model = YOLO("yolov10m.pt") 
model.export(format="onnx", imgsz=[480,640])
```

# Original YOLOv8 model
The original YOLOv10 model can be found in this repository: [YOLOv10 Repository](https://github.com/THU-MIG/yolov10)
- The License of the models is AGPL-3.0 license: [License](https://github.com/THU-MIG/yolov10/blob/main/LICENSE)

# Examples

 * **Image inference**:
 ```shell
 python image_object_detection.py
 ```

 * **Webcam inference**:
 ```shell
 python webcam_object_detection.py
 ```

 * **Video inference**: https://youtu.be/JShJpg8Mf7M
 ```shell
 python video_object_detection.py
 ```

 ![!YOLOv10 detection video](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/raw/main/doc/img/yolov8_video.gif)

  *Original video: [https://youtu.be/Snyg0RqpVxY](https://youtu.be/Snyg0RqpVxY)*

# References:
* YOLOv10 model: [https://github.com/THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
* YOLOv8 model: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* YOLOv5 model: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* YOLOv6 model: [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
* YOLOv7 model: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* PINTO0309's model zoo: [https://github.com/PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
* PINTO0309's model conversion tool: [https://github.com/PINTO0309/openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow)
