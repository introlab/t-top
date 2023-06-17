import argparse
import time

from PIL import Image, ImageDraw

import torch

from object_detection.modules.yolo_v4 import YoloV4
from object_detection.modules.yolo_v4_tiny import YoloV4Tiny
from object_detection.modules.yolo_v7 import YoloV7
from object_detection.modules.yolo_v7_tiny import YoloV7Tiny
from object_detection.datasets.coco_detection_transforms import CocoDetectionValidationTransforms
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CLASSES_INDEX

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def main():
    parser = argparse.ArgumentParser(description='Test the specified converted model')
    parser.add_argument('--model_type', choices=['yolo_v4', 'yolo_v4_tiny', 'yolo_v7', 'yolo_v7_tiny'],
                        help='Choose the mode', required=True)
    parser.add_argument('--weights_path', type=str, help='Choose the weights file path', required=True)
    parser.add_argument('--image_path', type=str, help='Choose the image file', required=True)

    args = parser.parse_args()

    model = create_model(args.model_type)
    model.load_weights(args.weights_path)
    model.eval()

    image = Image.open(args.image_path)

    predictions, scale = get_predictions(model, image)
    display_predictions(predictions, scale, image)


def create_model(model_type):
    if model_type == 'yolo_v4':
        return YoloV4()
    elif model_type == 'yolo_v4_tiny':
        return YoloV4Tiny()
    elif model_type == 'yolo_v7':
        return YoloV7()
    elif model_type == 'yolo_v7_tiny':
        return YoloV7Tiny()
    else:
        raise ValueError('Invalid model type')


def get_predictions(model, image):
    with torch.no_grad():
        transforms = CocoDetectionValidationTransforms(model.get_image_size(), one_hot_class=True)
        image_tensor, _, metadata = transforms(image, None)

        start = time.time()
        predictions = model(image_tensor.unsqueeze(0))
        print('Inference time: ', time.time() - start, 's')

        start = time.time()
        predictions = group_predictions(predictions)[0]
        predictions = filter_yolo_predictions(predictions, confidence_threshold=0.5, nms_threshold=0.45)
        print('Postprocessing time: ', time.time() - start, 's')

        return predictions, metadata['scale']


def display_predictions(predictions, scale, image):
    draw = ImageDraw.Draw(image)

    for prediction in predictions:
        center_x = prediction[X_INDEX].item() / scale
        center_y = prediction[Y_INDEX].item() / scale
        w = prediction[W_INDEX].item() / scale
        h = prediction[H_INDEX].item() / scale
        class_index = torch.argmax(prediction[CLASSES_INDEX:]).item()

        x0 = center_x - w / 2
        y0 = center_y - h / 2
        x1 = center_x + w / 2
        y1 = center_y + h / 2

        draw.rectangle([x0, y0, x1, y1], outline='red')
        draw.text((x0, y0), COCO_CLASSES[class_index], fill='red')

    del draw
    image.show()


if __name__ == '__main__':
    main()
