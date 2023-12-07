# yolo_performance

This folder folder contains the nodes to measure the performance of YOLO models.

## How to use

1. Capture images by executing this command and following the printed instructions.
```
roslaunch yolo_performance image_gathering.launch output_path:=<OUTPUT_PATH> dataset:=<coco or objects365> image_count:=<IMAGE_COUNT_PER_CLASS> camera_type:=<realsense, camera_2d_wide or "">
```

2. Process the images by executing this command.
```
roslaunch yolo_performance processing.launch input_path:=<INPUT_PATH> dataset:=<coco or objects365> neural_network_inference_type:=<cpu, torch_gpu or trt_gpu>
```
