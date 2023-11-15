import argparse
import time

from PIL import Image, ImageDraw

import torch

from object_detection.descriptor_yolo_v7 import DescriptorYoloV7
from object_detection.modules.yolo_v4 import YoloV4
from object_detection.modules.yolo_v4_tiny import YoloV4Tiny
from object_detection.modules.yolo_v7 import YoloV7
from object_detection.modules.yolo_v7_tiny import YoloV7Tiny
from object_detection.datasets.coco_detection_transforms import CocoDetectionValidationTransforms
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CLASSES_INDEX

from train_descriptor_yolo import _get_class_count

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

OBJECTS365_CLASS_NAMES = ['person', 'sneakers', 'chair', 'other shoes', 'hat', 'car', 'lamp', 'glasses', 'bottle',
                          'desk', 'cup', 'street lights', 'cabinet/shelf', 'handbag', 'bracelet', 'plate',
                          'picture/frame', 'helmet', 'book', 'gloves', 'storage box', 'boat', 'leather shoes',
                          'flower', 'bench', 'pottedplant', 'bowl', 'flag', 'pillow', 'boots', 'vase',
                          'microphone', 'necklace', 'ring', 'suv', 'wine glass', 'belt', 'tvmonitor', 'backpack',
                          'umbrella', 'traffic light', 'speaker', 'watch', 'tie', 'trash bin can', 'slippers',
                          'bicycle', 'stool', 'barrel/bucket', 'van', 'couch', 'sandals', 'basket', 'drum',
                          'pen/pencil', 'bus', 'bird', 'high heels', 'motorbike', 'guitar', 'carpet',
                          'cell phone', 'bread', 'camera', 'canned', 'truck', 'traffic cone', 'cymbal', 'lifesaver',
                          'towel', 'stuffed toy', 'candle', 'sailboat', 'laptop', 'awning', 'bed', 'faucet', 'tent',
                          'horse', 'mirror', 'power outlet', 'sink', 'apple', 'air conditioner', 'knife',
                          'hockey stick', 'paddle', 'pickup truck', 'fork', 'traffic sign', 'balloon', 'tripod', 'dog',
                          'spoon', 'clock', 'pot', 'cow', 'cake', 'dining table', 'sheep', 'hanger',
                          'blackboard/whiteboard', 'napkin', 'other fish', 'orange', 'toiletry', 'keyboard',
                          'tomato', 'lantern', 'machinery vehicle', 'fan', 'green vegetables', 'banana',
                          'baseball glove', 'aeroplane', 'mouse', 'train', 'pumpkin', 'soccer', 'skis', 'luggage',
                          'nightstand', 'tea pot', 'telephone', 'trolley', 'head phone', 'sports car', 'stop sign',
                          'dessert', 'scooter', 'stroller', 'crane', 'remote', 'refrigerator', 'oven', 'lemon', 'duck',
                          'baseball bat', 'surveillance camera', 'cat', 'jug', 'broccoli', 'piano', 'pizza',
                          'elephant', 'skateboard', 'surfboard', 'gun', 'skating and skiing shoes', 'gas stove',
                          'donut', 'bow tie', 'carrot', 'toilet', 'kite', 'strawberry', 'other balls', 'shovel',
                          'pepper', 'computer box', 'toilet paper', 'cleaning products', 'chopsticks', 'microwave',
                          'pigeon', 'baseball', 'cutting/chopping board', 'coffee table', 'side table', 'scissors',
                          'marker', 'pie', 'ladder', 'snowboard', 'cookies', 'radiator', 'fire hydrant', 'basketball',
                          'zebra', 'grape', 'giraffe', 'potato', 'sausage', 'tricycle', 'violin', 'egg',
                          'fire extinguisher', 'candy', 'fire truck', 'billiards', 'converter', 'bathtub',
                          'wheelchair', 'golf club', 'suitcase', 'cucumber', 'cigar/cigarette', 'paint brush', 'pear',
                          'heavy truck', 'hamburger', 'extractor', 'extension cord', 'tong', 'tennis racket',
                          'folder', 'american football', 'earphone', 'mask', 'kettle', 'tennis', 'ship', 'swing',
                          'coffee machine', 'slide', 'carriage', 'onion', 'green beans', 'projector', 'frisbee',
                          'washing machine/drying machine', 'chicken', 'printer', 'watermelon', 'saxophone', 'tissue',
                          'toothbrush', 'ice cream', 'hot-air balloon', 'cello', 'french fries', 'scale', 'trophy',
                          'cabbage', 'hot dog', 'blender', 'peach', 'rice', 'wallet/purse', 'volleyball', 'deer',
                          'goose', 'tape', 'tablet', 'cosmetics', 'trumpet', 'pineapple', 'golf ball', 'ambulance',
                          'parking meter', 'mango', 'key', 'hurdle', 'fishing rod', 'medal', 'flute', 'brush',
                          'penguin', 'megaphone', 'corn', 'lettuce', 'garlic', 'swan', 'helicopter', 'green onion',
                          'sandwich', 'nuts', 'speed limit sign', 'induction cooker', 'broom', 'trombone', 'plum',
                          'rickshaw', 'goldfish', 'kiwi fruit', 'router/modem', 'poker card', 'toaster', 'shrimp',
                          'sushi', 'cheese', 'notepaper', 'cherry', 'pliers', 'cd', 'pasta', 'hammer', 'cue',
                          'avocado', 'hamimelon', 'flask', 'mushroom', 'screwdriver', 'soap', 'recorder', 'bear',
                          'eggplant', 'board eraser', 'coconut', 'tape measure/ruler', 'pig', 'showerhead', 'globe',
                          'chips', 'steak', 'crosswalk sign', 'stapler', 'camel', 'formula 1', 'pomegranate',
                          'dishwasher', 'crab', 'hoverboard', 'meat ball', 'rice cooker', 'tuba', 'calculator',
                          'papaya', 'antelope', 'parrot', 'seal', 'butterfly', 'dumbbell', 'donkey', 'lion', 'urinal',
                          'dolphin', 'electric drill', 'hair dryer', 'egg tart', 'jellyfish', 'treadmill', 'lighter',
                          'grapefruit', 'game board', 'mop', 'radish', 'baozi', 'target', 'french', 'spring rolls',
                          'monkey', 'rabbit', 'pencil case', 'yak', 'red cabbage', 'binoculars', 'asparagus', 'barbell',
                          'scallop', 'noddles', 'comb', 'dumpling', 'oyster', 'table tennis paddle',
                          'cosmetics brush/eyeliner pencil', 'chainsaw', 'eraser', 'lobster', 'durian', 'okra',
                          'lipstick', 'cosmetics mirror', 'curling', 'table tennis']

CLASSES_BY_DATASET_TYPE = {'coco': COCO_CLASSES, 'objects365': OBJECTS365_CLASS_NAMES}


def main():
    parser = argparse.ArgumentParser(description='Test the specified converted model')
    parser.add_argument('--dataset_type', choices=['coco', 'objects365'], help='Choose the dataset type', required=True)
    parser.add_argument('--model_type', choices=['yolo_v4', 'yolo_v4_tiny', 'yolo_v7', 'yolo_v7_tiny',
                                                 'descriptor_yolo_v7'],
                        help='Choose the mode', required=True)
    parser.add_argument('--embedding_size', type=int, help='Choose the embedding size for descriptor_yolo_v7')
    parser.add_argument('--weights_path', type=str, help='Choose the weights file path', required=True)
    parser.add_argument('--image_path', type=str, help='Choose the image file', required=True)

    args = parser.parse_args()

    model = create_model(args.model_type, args.dataset_type, embedding_size=args.embedding_size)
    model.load_weights(args.weights_path)
    model.eval()

    image = Image.open(args.image_path)

    predictions, scale, offset_x, offset_y = get_predictions(model, image)
    display_predictions(predictions, scale, offset_x, offset_y, image, CLASSES_BY_DATASET_TYPE[args.dataset_type])


def create_model(model_type, dataset_type, embedding_size=None, class_probs=False):
    class_count = _get_class_count(dataset_type)
    if model_type == 'yolo_v4':
        model = YoloV4(class_count, class_probs=class_probs)
    elif model_type == 'yolo_v4_tiny':
        model = YoloV4Tiny(class_count, class_probs=class_probs)
    elif model_type == 'yolo_v7':
        model = YoloV7(dataset_type, class_probs=class_probs)
    elif model_type == 'yolo_v7_tiny':
        model = YoloV7Tiny(dataset_type, class_probs=class_probs)
    elif model_type == 'descriptor_yolo_v7':
        model = DescriptorYoloV7(class_count, embedding_size=embedding_size, class_probs=class_probs)
    else:
        raise ValueError('Invalid model type')

    return model


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

        return predictions, metadata['scale'], metadata['offset_x'], metadata['offset_y']


def display_predictions(predictions, scale, offset_x, offset_y, image, classses):
    draw = ImageDraw.Draw(image)

    for prediction in predictions:
        center_x = (prediction[X_INDEX].item() - offset_x) / scale
        center_y = (prediction[Y_INDEX].item() - offset_y) / scale
        w = prediction[W_INDEX].item() / scale
        h = prediction[H_INDEX].item() / scale
        class_index = torch.argmax(prediction[CLASSES_INDEX:]).item()

        x0 = center_x - w / 2
        y0 = center_y - h / 2
        x1 = center_x + w / 2
        y1 = center_y + h / 2

        draw.rectangle([x0, y0, x1, y1], outline='red')
        draw.text((x0, y0), classses[class_index], fill='red')

    del draw
    image.show()


if __name__ == '__main__':
    main()
