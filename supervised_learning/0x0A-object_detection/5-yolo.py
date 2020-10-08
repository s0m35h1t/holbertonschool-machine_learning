#!/usr/bin/env python3
"""Define: Yolo class """

import tensorflow.keras as K
import numpy as np
import cv2
import os


def iou(box_a, box_b):
    """intersection over union
    Args:
        box_a
        box_b
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return area / (a_area + b_area - area)


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
        sort_order = np.lexsort((-box_scores, box_classes))
        box_scores = box_scores[sort_order]
        box_classes = box_classes[sort_order]
        filtered_boxes = filtered_boxes[sort_order]
        del_idxs = []
        for idx in range(len(box_scores)):
            if idx in del_idxs:
                continue
            clas = box_classes[idx]
            box = filtered_boxes[idx]
            for cidx in range(idx + 1, len(box_scores)):
                if (box_classes[cidx] != clas):
                    break
                if ((iou(filtered_boxes[cidx], box)
                     >= self.nms_t)):
                    del_idxs.append(cidx)
        return (np.delete(filtered_boxes, del_idxs, axis=0),
                np.delete(box_classes, del_idxs),
                np.delete(box_scores, del_idxs))

    @staticmethod
    def load_images(folder_path):
        """load images from folder
        Args:
            folder_path: a string representing the path
            to the folder holding all the images to load
        Returns
            tuple of (images, image_paths):

                images: a list of images as
                numpy.ndarrays
                image_paths: a list of paths to the
                individual images in images
        """
        fp_list = os.listdir(folder_path)
        imgs = []
        file_paths = []
        for f in fp_list:
            path = folder_path + '/' + f
            imgs.append(cv2.imread(folder_path + '/' + f))
            file_paths.append(path)
        return imgs, file_paths

    def preprocess_images(self, images):
        """Resize images
        Args:
            images: a list of images as numpy.ndarrays
        Returns:
                tuple of (pimages, image_shapes):

            pimages: a numpy.ndarray of shape
                (ni, input_h, input_w, 3) containing
                all of the preprocessed images
                ni: the number of images that were
                preprocessed
                input_h: the input height for the Darknet model Note: this can vary by model
                input_w: the input width for the Darknet model Note: this can vary by model
                3: number of color channels
            image_shapes: a numpy.ndarray of shape
                (ni, 2) containing the original height
                and width of the images
                2 => (image_height, image_width)
"""
        image_shapes = np.empty((len(images), 2))
        pimages = np.empty((len(images), input_h, intput_w, 3))
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        for i, im in enumerate(images):
            image_shapes[i][0] = im.shape[0]
            image_shapes[i][1] = im.shape[1]
            pimages[i] = cv2.resize(im / 255,
                                    (input_h, input_w),
                                    interpolation=cv2.INTER_CUBIC)
        return pimages, image_shapes
