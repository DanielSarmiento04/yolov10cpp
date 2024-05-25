# yolov10


## Prepare the code 


1. Download de model you want 

  
  - yolov10n
  - yolov10s
  - yolov10m
  - yolov10b
  - yolov10l
  - yolov10x


```bash
    python download_model.py  --model {MODEL_SELECTED}
```

## Convert model

```
    yolo export model=yolov10n.pt format=onnx
```


## How to run this code 


1. Using Cmake, Recommended

```
    mkdir build
    cd build
    cmake ..
    make
```

