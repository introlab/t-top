import torch

from object_detection.criterions.yolo_v4_loss import calculate_iou
from object_detection.modules.yolo_layer import CONFIDENCE_INDEX, CLASSES_INDEX


def group_predictions(predictions):
    grouped_predictions = []
    for i in range(len(predictions)):
        N = predictions[i].size()[0]
        H = predictions[i].size()[1]
        W = predictions[i].size()[2]
        N_ANCHORS = predictions[i].size()[3]
        N_PREDICTION = predictions[i].size()[4]
        grouped_predictions.append(predictions[i].reshape(N, H * W * N_ANCHORS, N_PREDICTION))

    return torch.cat(grouped_predictions, dim=1)


def filter_yolo_predictions(predictions, confidence_threshold=0.7, nms_threshold=0.6):
    predictions = predictions[predictions[:, CONFIDENCE_INDEX] > confidence_threshold]
    return _nms(predictions, nms_threshold)


def filter_yolo_predictions_by_classes(predictions, confidence_threshold=0.7, nms_threshold=0.6):
    predictions = predictions[predictions[:, CONFIDENCE_INDEX] > confidence_threshold]
    predictions_by_class_index = _group_predictions_by_class_index(predictions)

    valid_predictions = []
    for c, p in predictions_by_class_index.items():
        valid_predictions += _nms(p, nms_threshold)
    return valid_predictions


def _group_predictions_by_class_index(predictions):
    class_count = predictions[:, CLASSES_INDEX:].size(1)
    class_indexes = torch.argmax(predictions[:, CLASSES_INDEX:], dim=1).tolist()

    predictions_by_class_index = {c: [] for c in range(class_count)}
    for i, c in enumerate(class_indexes):
        predictions_by_class_index[c].append(predictions[i:i + 1])

    tensor_predictions_by_class_index = {}
    for c in predictions_by_class_index.keys():
        if len(predictions_by_class_index[c]) > 0:
            tensor_predictions_by_class_index[c] = torch.cat(predictions_by_class_index[c], dim=0)

    return tensor_predictions_by_class_index


def _nms(predictions, nms_threshold):
    sorted_index = torch.argsort(predictions[:, CONFIDENCE_INDEX], descending=True)
    predictions = predictions[sorted_index]

    valid_predictions = []

    while predictions.size(0) > 0:
        valid_predictions.append(predictions[0])
        predictions = predictions[1:]

        if predictions.size(0) == 0:
            break

        iou = calculate_iou(valid_predictions[-1][:CONFIDENCE_INDEX].repeat(predictions.size()[0], 1),
                            predictions[:, :CONFIDENCE_INDEX])

        predictions = predictions[iou < nms_threshold]

    return valid_predictions
