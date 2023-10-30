This is a tutorial about training the yolov7 model to recognize rear signal light and tell braking and normal status. It's credit to: [WongKinYiu](https://github.com/WongKinYiu/yolov7.git) and [Armaan Sandhu](https://medium.com/@armaan.sandhu.2002/training-yolov7-to-detect-vehicle-braking-e8e7e9db1b3b#3a49)

The following steps can be operated on google colab

## Step 1 Creating training enviroment
1.1 Mount Google Drive
``` shell
from google.colab import drive
drive.mount('/content/gdrive')
```
1.2 Download YOLOv7 repository
``` shell
!git clone https://github.com/WongKinYiu/yolov7.git
```
1.3 Install YOLOv7 dependencies
``` shell
%cd yolov7
!pip install -r requirements.txt
```
1.4 Download pretrained model
``` shell
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## Step 2 Dataset Preprocessing and Preparation

First, you have to download the already-prepared dataset from [here]()
2.1

![output_3](https://github.com/HunterWang123456/yolov7_for_braking/assets/74261517/4fc41c1b-00a4-40d2-963e-f6d14ad4ed46)

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
