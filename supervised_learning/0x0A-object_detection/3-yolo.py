#!/usr/bin/env python3
"""Define: Yolo class """

import tensorflow.keras as K
import numpy as np
import cv2
import os


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
            containing the processed boundary boxes for each output,
            respectively
            box_confidences: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output,
            respectively
        box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
                containing the processed box class probabilities
                for each output, respectively
        Returns:
            tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes: a numpy.ndarray of shape (?, 4)
                containing all of the filtered bounding boxes:
                box_classes: a numpy.ndarray of shape (?,)
                containing the class number that each box in
                filtered_boxes predicts, respectively
                box_scores: a numpy.ndarray of shape (?)
                containing the box scores for each box in filtered_boxes,
                respectively

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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non max suppression on boxes
        Args:
            filtered_boxes: a numpy.ndarray of shape (?, 4)
                containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,)
                containing the class number for the class that
                filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?)
                containing the box scores for each box
                in filtered_boxes, respectively
        Returns:
            tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores):

                box_predictions: a numpy.ndarray of shape (?, 4)
                    containing all of the predicted bounding boxes
                    ordered by class and box score
                predicted_box_classes: a numpy.ndarray of shape (?,)
                    containing the class number for box_predictions ordered
                    by class and box score, respectively
                predicted_box_scores: a numpy.ndarray of shape (?)
                    containing the box scores for box_predictions ordered
                    by class and box score, respectively
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)

            filtered = filtered_boxes[index]
            scores = box_scores[index]
            classe = box_classes[index]

            x1 = filtered[:, 0]
            x2 = filtered[:, 2]
            y1 = filtered[:, 1]
            y2 = filtered[:, 3]

            keep = []
            area = (x2 - x1) * (y2 - y1)
            index_list = np.flip(scores.argsort(), axis=0)

            while len(index_list) > 0:
                pos1 = index_list[0]
                pos2 = index_list[1:]
                keep.append(pos1)

                xx1 = np.maximum(x1[pos1], x1[pos2])
                yy1 = np.maximum(y1[pos1], y1[pos2])
                xx2 = np.minimum(x2[pos1], x2[pos2])
                yy2 = np.minimum(y2[pos1], y2[pos2])

                height = np.maximum(0.0, yy2 - yy1)
                width = np.maximum(0.0, xx2 - xx1)

                intersection = (width * height)
                union = area[pos1] + area[pos2] - intersection
                iou = intersection / union
                below_threshold = np.where(iou <= self.nms_t)[0]
                index_list = index_list[below_threshold + 1]

            keep = np.array(keep)

            box_predictions.append(filtered[keep])
            predicted_box_classes.append(classe[keep])
            predicted_box_scores.append(scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)
