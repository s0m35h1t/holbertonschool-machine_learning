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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter box outputs
        Agrs:
        boxes: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output, respectively
            box_confidences: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
                containing the processed box class probabilities
                for each output, respectively
        Returns:
            tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes: a numpy.ndarray of shape (?, 4)
                containing all of the filtered bounding boxes:
                box_classes: a numpy.ndarray of shape (?,)
                containing the class number that each box in filtered_boxes predicts, respectively
                box_scores: a numpy.ndarray of shape (?)
                containing the box scores for each box in filtered_boxes, respectively

        """
        _boxes = np.concatenate([boxs.reshape(-1, 4) for boxs in boxes])
        class_probs = np.concatenate([probs.reshape(-1,
                                                    box_class_probs[0].
                                                    shape[-1])
                                      for probs in box_class_probs])
        _classes = class_probs.argmax(axis=1)
        _confidences = (np.concatenate([conf.reshape(-1)
                                        for conf in box_confidences])
                        * class_probs.max(axis=1))
        thresh_idxs = np.where(_confidences < self.class_t)
        return (np.delete(_boxes, thresh_idxs, axis=0),
                np.delete(_classes, thresh_idxs),
                np.delete(_confidences, thresh_idxs))
