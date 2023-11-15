import copy

import torchvision.datasets as datasets

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_COUNT = 80


class ObjectDetectionCoco(datasets.CocoDetection):
    def __init__(self, root, ann_file, transforms=None):
        super(ObjectDetectionCoco, self).__init__(root, ann_file)
        self._ann_file = ann_file

        if transforms is None:
            raise ValueError('Invalid transforms')
        self._transforms = transforms

    def __getitem__(self, index):
        image, target = super(ObjectDetectionCoco, self).__getitem__(index)
        image_id = self.ids[index]

        initial_width, initial_height = image.size

        target = copy.deepcopy(target)
        image, target, transforms_metadata = self._transforms(image, target)
        metadata = {
            'image_id': image_id,
            'initial_width': initial_width,
            'initial_height': initial_height,
            'scale': transforms_metadata['scale'],
            'offset_x': transforms_metadata['offset_x'],
            'offset_y': transforms_metadata['offset_y']
        }
        return image, target, metadata

    def evaluate(self, result_file):
        coco_gt = COCO(self._ann_file)
        coco_dt = coco_gt.loadRes(result_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        names = ['AP', 'Ap .5', 'AP .75', 'AP (S)', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (S)', 'AR (M)',
                 'AR (L)']

        return coco_eval.stats, names
