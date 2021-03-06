#!/usr/bin/env python3
"""Define: Yolo class """

import tensorflow.keras as K
import numpy as np


class Yolo:
    """YOLO class"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        Args:
            outputs
            image_size
        Returns:
            Returns a tuple of
            (boxes, box_confidences, box_class_probs)
        """
        boxes = [output[:, :, :, 0:4] for output in outputs]
        for oidx, output in enumerate(boxes):
            for y in range(output.shape[0]):
                for x in range(output.shape[1]):
                    centery = ((1 / (1 + np.exp(-output[y, x, :, 1])) + y)
                               / output.shape[0] * image_size[0])
                    centerx = ((1 / (1 + np.exp(-output[y, x, :, 0])) + x)
                               / output.shape[1] * image_size[1])
                    p_resizes = self.anchors[oidx].astype(float)
                    p_resizes[:, 0] *= (np.exp(output[y, x, :, 2])
                                        / 2 * image_size[1] /
                                        self.model.input.shape[1].value)
                    p_resizes[:, 1] *= (np.exp(output[y, x, :, 3])
                                        / 2 * image_size[0] /
                                        self.model.input.shape[2].value)
                    output[y, x, :, 0] = centerx - p_resizes[:, 0]
                    output[y, x, :, 1] = centery - p_resizes[:, 1]
                    output[y, x, :, 2] = centerx + p_resizes[:, 0]
                    output[y, x, :, 3] = centery + p_resizes[:, 1]
        box_confidences = [1 / (1 + np.exp(-output[:, :, :, 4, np.newaxis]))
                           for output in outputs]
        box_class_probs = [1 / (1 + np.exp(-output[:, :, :, 5:]))
                           for output in outputs]
        return boxes, box_confidences, box_class_probs
