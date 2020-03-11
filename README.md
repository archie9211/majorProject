# majorProject

Our Implementation of "Abnormal Vehicle Detection System for Intelligent
Transportation" 

Steps and what we used for the same :

* Vehicle detection  | yolov3 implemented by https://github.com/zzh8829/yolov3-tf2
* Optical Flow calculation | Farenbeck Method
* Classification | SVM

## Usage

clone the repo 
```
https://github.com/archie9211/majorProject/
cd majorProject
```

then download the yolo weights
```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py
```

Running simple yolo on a video on webcam (--video 0) 

there are three modes of execution :
```
basic : only yolo object detection 
optical_flow : detect object and calculate optical flow on detected objects
aggregated : show only aggregated optical flow vector of object
final : classify the object into safe(green) or unsafe(red)
```

```
python main.py --video /path/to/video --mode basic/optical_flow/aggregated/final 
```

run following command for full help of arguments and their uses :
```
python main.py --help
```


### Contributers :

    Nageen Chand (16MI523) 
    Akash Soni (16MI529)
    Sachi Doegar (16MI512)
    Department of Computer Science and Engineering
    National Institute of Technology Hamirpur (H.P), 177005
