import os
import sys

import torch
import torch.nn.functional as F

from dnn_utils.dnn_model import PACKAGE_PATH, DnnModel

sys.path.append(os.path.join(PACKAGE_PATH, '..', '..', '..', 'tools', 'dnn_training'))
from object_detection.modules.yolo_layer import X_INDEX, Y_INDEX, W_INDEX, H_INDEX, CONFIDENCE_INDEX, CLASSES_INDEX
from object_detection.filter_yolo_predictions import group_predictions, filter_yolo_predictions_by_classes


IMAGE_SIZE_BY_MODEL_NAME = {
    'yolo_v4_coco' : (608, 608),
    'yolo_v4_tiny_coco' : (416, 416),
    'yolo_v7_coco' : (640, 640),
    'yolo_v7_tiny_coco' : (640, 640),
    'yolo_v7_objects365' : (640, 640),
}
IN_CHANNELS = 3

COCO_CLASS_NAMES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'dining table',
                    'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair dryer',
                    'toothbrush']

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

CLASS_NAMES_BY_MODEL_NAME = {
    'yolo_v4_coco' : COCO_CLASS_NAMES,
    'yolo_v4_tiny_coco' : COCO_CLASS_NAMES,
    'yolo_v7_coco' : COCO_CLASS_NAMES,
    'yolo_v7_tiny_coco' : COCO_CLASS_NAMES,
    'yolo_v7_objects365' : OBJECTS365_CLASS_NAMES,
}


class YoloPrediction:
    def __init__(self, prediction_tensor, scale, offset_x, offset_y, class_count):
        self.center_x = ((prediction_tensor[X_INDEX] - offset_x) / scale).item()
        self.center_y = ((prediction_tensor[Y_INDEX] - offset_y) / scale).item()
        self.width = (prediction_tensor[W_INDEX] / scale).item()
        self.height = (prediction_tensor[H_INDEX] / scale).item()
        self.confidence = prediction_tensor[CONFIDENCE_INDEX].item()

        class_probabilities =  prediction_tensor[CLASSES_INDEX:CLASSES_INDEX + class_count]
        self.class_index = torch.argmax(class_probabilities, dim=0).item()
        self.class_probabilities = class_probabilities.tolist()

        self.descriptor = []


class Yolo(DnnModel):
    def __init__(self, model_name, confidence_threshold=0.99, nms_threshold=0.5, inference_type=None):
        if model_name not in IMAGE_SIZE_BY_MODEL_NAME:
            raise ValueError(f'Invalid model name ({model_name})')

        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

        self._image_size = IMAGE_SIZE_BY_MODEL_NAME[model_name]

        torch_script_model_path = os.path.join(PACKAGE_PATH, 'models', f'{model_name}.ts.pth')
        tensor_rt_model_path = os.path.join(PACKAGE_PATH, 'models', f'{model_name}.trt.pth')
        sample_input = torch.ones(1, 3, self._image_size[0], self._image_size[1])

        super(Yolo, self).__init__(torch_script_model_path, tensor_rt_model_path, sample_input,
                                   inference_type=inference_type)
        self._padded_image = torch.ones(1, 3, self._image_size[0], self._image_size[1]).to(self._device)

    def get_supported_image_size(self):
        return IMAGE_SIZE_BY_MODEL_NAME[self._model_name]

    def get_class_names(self):
        return CLASS_NAMES_BY_MODEL_NAME[self._model_name]

    def __call__(self, image_tensor):
        with torch.no_grad():
            scale, offset_x, offset_y, predictions = self.forward_raw(image_tensor.to(self._device))
            predictions = group_predictions(predictions)[0]
            predictions = filter_yolo_predictions_by_classes(predictions,
                                                             confidence_threshold=self._confidence_threshold,
                                                             nms_threshold=self._nms_threshold)

            class_count = len(CLASS_NAMES_BY_MODEL_NAME[self._model_name])
            return [YoloPrediction(prediction.cpu(), scale, offset_x, offset_y, class_count) for prediction in predictions]

    def forward_raw(self, image_tensor):
        scale, offset_x, offset_y = self._set_image(image_tensor.to(self._device).unsqueeze(0))
        predictions = super(Yolo, self).__call__(self._padded_image)
        return scale, offset_x, offset_y, predictions

    def _set_image(self, image_tensor):
        scale = min(self._padded_image.size()[2] / image_tensor.size()[2], self._padded_image.size()[3] / image_tensor.size()[3])
        output_size = ((int(image_tensor.size()[2] * scale), int(image_tensor.size()[3] * scale)))
        offset_y = int((self._padded_image.size()[2] - output_size[0]) / 2)
        offset_x = int((self._padded_image.size()[3] - output_size[1]) / 2)

        image_tensor = F.interpolate(image_tensor, size=output_size, mode='bilinear')
        self._padded_image[:] = 0.44705882352
        self._padded_image[:, :, offset_y:offset_y + image_tensor.size()[2], offset_x:offset_x + image_tensor.size()[3]] = image_tensor
        return scale, offset_x, offset_y
