# dnn_utils

This ROS package contains the Python classes to use the trained neural networks
from [dnn_training](../../tools/dnn_training). The neural networks are executed on the CPU, GPU with PyTorch or GPU with
TensorRT depending on the executing speed.

## Neural Network Class Descriptions

### `AudioDescriptorExtractor`

This neural network classifies a sound to one class and extracts an embedding. It takes as input a mono audio frame.

### `DescriptorYolo`

This neural network detects objects, classifies them and extracts embeddings for them. It takes as input a RGB image.
The model is a modified version of `YOLOv4-tiny (COCO)` or `YOLOv7 (COCO)`.
`YOLOv7 (COCO)` is training with [this repository](https://github.com/mamaheux/descriptor-yolov7).

### `FaceDescriptorExtractor`

This neural network extracts an embedding for a face. It takes as input an image containing a person and the person's
pose.

### `MulticlassAudioDescriptorExtractor`

This neural network classifies a sound to many classes and extracts an embedding. It takes as input a mono audio frame.

### `PoseEstimator`

This neural network estimates the pose of a person. It takes as input an image containing a person.

### `VoiceDescriptorExtractor`

This neural network extracts an embedding for a sound containing a voice. It takes as input a mono audio frame
containing a voice.

### `Yolo`

This neural network detects objects and classifies them. It takes as input a RGB image. The model is `YOLOv4 (COCO)`,
`YOLOv4-tiny (COCO)`, `YOLOv7 (COCO)`, `YOLOv7-tiny (COCO)` or `YOLOv7 (Objects365)`.
The pre-trained weights are used except for `YOLOv7 (Objects365)` which is trained with [this repository](https://github.com/WongKinYiu/yolov7).
