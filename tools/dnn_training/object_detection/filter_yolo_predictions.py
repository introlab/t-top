import torch

from object_detection.criterions.yolo_v4_loss import calculate_iou
from object_detection.modules.yolo_layer import CONFIDENCE_INDEX


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
