# DNN Training
This folder contains the tools to train the neural networks used by T-Top.

## Folder Structure
- Training scripts
    - The [train_audio_descriptor_extractor.py](train_audio_descriptor_extractor.py) script trains a neural network that classifies a sound to one class and extracts an embedding. The network name is `audio_descriptor_extractor`.
    - The [train_descriptor_yolo_v4.py](train_descriptor_yolo_v4.py) script trains a neural network that detects objects, classifies them and extracts embeddings. The network name is `descriptor_yolo_v4`.
    - The [train_face_descriptor_extractor.py](train_face_descriptor_extractor.py) script trains a neural network that extracts embeddings for a face. The network name is `face_descriptor_extractor`.
    - The [train_keyword_spotter.py](train_keyword_spotter.py) script trains a neural network that detects a wake-up word. The network name is `keyword_spotter`.
    - The [train_multiclass_audio_descriptor_extractor.py](train_multiclass_audio_descriptor_extractor.py) script trains a neural network that classifies a sound to many classes and extracts an embedding. The network name is `audio_descriptor_extractor`.
    - The [train_pose_estimator.py](train_pose_estimator.py) script trains a neural network that estimates the pose of a person. The network name is `pose_estimator`.
- Export scripts
    - The [export_audio_descriptor_extractor.py](export_audio_descriptor_extractor.py) script exports the `audio_descriptor_extractor` network to a TorchScript file and a TensorRT file.
    - The [export_descriptor_yolo_v4.py](export_descriptor_yolo_v4.py) script exports the `descriptor_yolo_v4` network to a TorchScript file and a TensorRT file.
    - The [export_face_descriptor_extractor.py](export_face_descriptor_extractor.py) script exports the `face_descriptor_extractor` network to a TorchScript file and a TensorRT file.
    - The [export_keyword_spotter.py](export_keyword_spotter.py) script exports the `keyword_spotter` network to a TorchScript file and a TensorRT file.
    - The [export_pose_estimator.py](export_pose_estimator.py) script exports the `pose_estimator` network to a TorchScript file and a TensorRT file.
    - The [export_yolo_v4.py](export_yolo_v4.py) script exports the `yolo_v4` network to a TorchScript file and a TensorRT file.
- Test scripts
    - The [test_exported_audio_descriptor_extractor.py](test_exported_audio_descriptor_extractor.py) script tests the exported `audio_descriptor_extractor` network.
    - The [test_exported_descriptor_yolo_v4.py](test_exported_descriptor_yolo_v4.py) script tests the exported `descriptor_yolo_v4` network.
    - The [test_exported_face_descriptor_extractor.py](test_exported_face_descriptor_extractor.py) script tests the exported `face_descriptor_extractor` network.
    - The [test_exported_keyword_spotter.py](test_exported_keyword_spotter.py) script tests the exported `keyword_spotter` network.
    - The [test_exported_pose_estimator.py](test_exported_pose_estimator.py) script tests the exported `pose_estimator` network.
    - The [test_lfw_vox_celeb_models.py](test_lfw_vox_celeb_models.py) script tests the combination of the `audio_descriptor_extractor` network and the `face_descriptor_extractor` network.

## Setup

1. Setup the Python virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy
pip install -r requirements.txt
```

2. Show the help of the script :
```bash
python script.py -h
```

3. Use the script according to the help :
```bash
python script.py ...
```
