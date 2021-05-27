import itertools

import numpy
import numpy as np

from collections import defaultdict, Counter

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize

from evalme.eval_item import EvalItem
from evalme.utils import get_text_comparator, texts_similarity, Result


class ObjectDetectionEvalItem(EvalItem):
    SHAPE_KEY = 'undefined'

    def _iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA['x'], boxB['x'])
        yA = max(boxA['y'], boxB['y'])
        xB = min(boxA['x'] + boxA['width'], boxB['x'] + boxB['width'])
        yB = min(boxA['y'] + boxA['height'], boxB['y'] + boxB['height'])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA['width'] * boxA['height']
        boxBArea = boxB['width'] * boxB['height']

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def total_iou(self, item, label_weights=None, algorithm=None, qval=None, per_label=False):
        """
        For each shape in current eval item, we compute IOU with identically labeled shape with largest intersection.
        This is suboptimal metric since it doesn't consider cases where multiple boxes from self coincides with
        with single box from item
        :param item: to be compared with self
        :param label_weights: weight of particular label
        :param algorithm: algorithm of comparing values
        :param qval: q value
        :param per_label: calculate per label or overall
        :return:
        """
        label_weights = label_weights or {}
        if per_label:
            ious = defaultdict(list)
        else:
            ious, weights = [], []
        comparator = get_text_comparator(algorithm, qval)
        for gt in self.get_values_iter():
            max_iou = 0
            for pred in item.get_values_iter():
                label_sim = texts_similarity(gt[self._shape_key], pred[self._shape_key], comparator)
                if label_sim == 0:
                    continue
                iou = self._iou(gt, pred)
                max_iou = max(iou, max_iou)
            if per_label:
                for l in gt[self._shape_key]:
                    ious[l].append(max_iou)
            else:
                weight = sum(label_weights.get(l, 1) for l in gt[self._shape_key])
                ious.append(max_iou * weight)
                weights.append(weight)
        if per_label:
            return {l: float(np.mean(v)) for l, v in ious.items()}
        return np.average(ious, weights=weights) if ious else 0.0

    def _precision_recall_at_iou_per_label(self, item, iou_threshold):
        """
        :return: precision dict {label: float[0..1]}
                 recall dict {label: float[0..1]}
        """
        shapes = item.get_values()
        tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)

        def inc_counters(c, labels):
            for l in labels:
                c[l] += 1

        all_labels = set()
        for shape_pred in self.get_values_iter():
            [all_labels.add(l) for l in shape_pred[self._shape_key]]
            for shape_gt in shapes:
                [all_labels.add(l) for l in shape_gt[self._shape_key]]
                iou = self._iou(shape_pred, shape_gt)
                if shape_pred[self._shape_key] == shape_gt[self._shape_key]:
                    if iou >= iou_threshold:
                        # IOU > t with matching labels => true positive
                        inc_counters(tp, shape_gt[self._shape_key])
                    else:
                        # IOU < t with matching labels => false negative
                        inc_counters(fn, shape_gt[self._shape_key])
                else:
                    if iou >= iou_threshold:
                        # IOU > t with non-matching labels => false positive
                        inc_counters(fp, shape_gt[self._shape_key])

        precision, recall = {}, {}
        for l in all_labels:
            totalp = tp[l] + fp[l]
            total_true = tp[l] + fp[l]
            precision[l] = tp[l] / totalp if totalp > 0 else 0
            recall[l] = tp[l] / total_true if total_true > 0 else 0
        return precision, recall

    def _precision_recall_at_iou(self, item, iou_threshold, label_weights=None, per_label=False):

        if per_label:
            # it's identical except that it returns per-label scores rather than averaged
            return self._precision_recall_at_iou_per_label(item, iou_threshold)

        shapes = item.get_values()
        tp, fp, fn = 0, 0, 0
        label_weights = label_weights or {}

        for shape_pred in self.get_values_iter():
            for shape_gt in shapes:
                iou = self._iou(shape_pred, shape_gt)
                if self._shape_key in shape_gt:
                    weight = sum(label_weights.get(l, 1) for l in shape_gt[self._shape_key])
                else:
                    weight = 1
                if shape_pred[self._shape_key] == shape_gt[self._shape_key]:
                    if iou >= iou_threshold:
                        tp += weight
                    else:
                        fn += weight
                else:
                    if iou >= iou_threshold:
                        fp += weight
        totalp = tp + fp
        total_true = tp + fn
        precision = tp / totalp if totalp > 0 else 0
        recall = tp / total_true if total_true > 0 else 0
        return precision, recall

    def prediction_result_at_iou_for_all_bbox(self, item, iou_threshold=0.5):
        """
        Prediction result at iou for all labels in bbox
        """
        pred = item.get_values()
        results = {}
        gt = self.get_values()

        if len(pred) > len(gt):
            return Result.FP
        if len(pred) < len(gt):
            return Result.FN
        else:
            gt_count = dict(Counter(list(itertools.chain.from_iterable([shape[self._shape_key] for shape in gt]))))
            pred_count = dict(Counter(list(itertools.chain.from_iterable([shape[self._shape_key] for shape in pred]))))
            for key in gt_count:
                if gt_count[key] > pred_count[key]:
                    return Result.FN
                if gt_count[key] < pred_count[key]:
                    return Result.FP

        for i in range(len(gt)):
            shape_gt = gt[i]
            max_iou = -1
            for shape_pred in pred:
                if shape_pred[self._shape_key] == shape_gt[self._shape_key]:
                    iou = self._iou(shape_gt, shape_pred)
                    max_iou = max(iou, max_iou)
            results[i] = max_iou

        if all(val >= iou_threshold for val in results.values()):
            return Result.TP
        if all(val >= 0 & val < iou_threshold for val in results.values()):
            return Result.FP
        if any(val == -1 for val in results.values()):
            return Result.FN

    def prediction_result_at_iou_for_all_bbox_per_label(self, item, iou_threshold=0.5):
        #TODO implement on demand
        pass

    def precision_at_iou(self, item, iou_threshold=0.5, label_weights=None, per_label=False):
        precision, _ = self._precision_recall_at_iou(item, iou_threshold, label_weights, per_label)
        return precision

    def recall_at_iou(self, item, iou_threshold=0.5, label_weights=None, per_label=False):
        _, recall = self._precision_recall_at_iou(item, iou_threshold, label_weights, per_label)
        return recall

    def f1_at_iou(self, item, iou_threshold=0.5, label_weights=None, per_label=False):
        precision, recall = self._precision_recall_at_iou(item, iou_threshold, label_weights, per_label)
        if per_label:
            out = {}
            for l in precision:
                denom = precision[l] + recall[l]
                if denom == 0:
                    out[l] = 0
                else:
                    out[l] = 2 * precision[l] * recall[l] / denom
            return out
        denom = precision + recall
        if denom == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    def mAP_at_iou(self, item, iou_threshold=0.5, label_weights=None, per_label=False):
        if per_label:
            precisions = defaultdict(list)
            recalls = defaultdict(list)
        else:
            precisions = []
            recalls = []
        thresholds = numpy.arange(start=0.2, stop=iou_threshold, step=0.05)
        for threshold in thresholds:
            precision, recall = self._precision_recall_at_iou(item, threshold, label_weights, per_label)
            if per_label:
                for label in precision:
                    precisions[label].append(precision[label])
                    recalls[label].append(recall[label])
            else:
                precisions.append(precision)
                recalls.append(recall)
        if per_label:
            for label in precisions:
                precisions[label].append(1)
                recalls[label].append(0)
        else:
            precisions.append(1)
            recalls.append(0)
        if per_label:
            AP = {}
            for label in precisions:
                rec = numpy.array(recalls[label])
                prec = numpy.array(precisions[label])
                AP[label] = numpy.sum((rec[:-1] - rec[1:]) * prec[:-1])
        else:
            recalls = numpy.array(recalls)
            precisions = numpy.array(precisions)
            AP = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        return AP

    def total_iou_matrix(self, item, label_weights=None, algorithm=None, qval=None, per_label=False):
        """
        For each shape in current eval item, we compute IOU with identically labeled shape.
        :param item: to be compared with self
        :param label_weights: weight of particular label
        :param algorithm: algorithm of comparing values
        :param qval: q value
        :param per_label: calculate per label or overall
        :return:
        """
        label_weights = label_weights or {}
        ious = []
        comparator = get_text_comparator(algorithm, qval)
        for gt in self.get_values_iter():
            for pred in item.get_values_iter():
                label_sim = texts_similarity(gt[self._shape_key], pred[self._shape_key], comparator)
                if label_sim == 0:
                    continue
                iou = self._iou(gt, pred)
                weight = sum(label_weights.get(l, 1) for l in gt[self._shape_key])
                result = dict()
                result['iou'] = iou * weight
                result['weight'] = weight
                result['prediction'] = pred
                result['groundtruth'] = gt
                ious.append(result)
        return ious


class BboxObjectDetectionEvalItem(ObjectDetectionEvalItem):
    SHAPE_KEY = 'rectanglelabels'


class PolygonObjectDetectionEvalItem(ObjectDetectionEvalItem):

    SHAPE_KEY = 'polygonlabels'

    def _area_close(self, p1, p2):
        return abs(p1.area / max(p2.area, 1e-8) - 1) < 0.1

    def _try_build_poly(self, points):
        poly = Polygon(points)
        # Everything is OK, points are valid polygon
        if poly.is_valid:
            return poly

        # Polygon contains small bowties, we can fix them:
        # (https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera)
        fixed_poly = poly.buffer(0)
        if not fixed_poly.geom_type == 'MultiPolygon' and self._area_close(fixed_poly, poly):
            return fixed_poly

        # Trying to build multiple polygons from self intersected polygon
        line_non_simple = LineString(points)
        mls = unary_union(line_non_simple)
        polygons = list(polygonize(mls))
        multi_poly = MultiPolygon(polygons)
        if len(polygons) and multi_poly.is_valid:
            return multi_poly

        # Trying to draw convex hull
        convex_hull = poly.convex_hull
        if self._area_close(convex_hull, poly):
            return convex_hull

        # We are failing to build polygon, this shall be reported via error log
        raise ValueError(f'Fail to build polygon from {points}')

    def _iou(self, polyA, polyB):
        pA = self._try_build_poly(polyA['points'])
        pB = self._try_build_poly(polyB['points'])
        inter_area = pA.intersection(pB).area
        iou = inter_area / (pA.area + pB.area - inter_area)
        return iou


def _as_bboxes(item, shape_key=None):
    if not isinstance(item, BboxObjectDetectionEvalItem):
        return BboxObjectDetectionEvalItem(item, shape_key)
    return item


def _as_polygons(item, shape_key=None):
    if not isinstance(item, PolygonObjectDetectionEvalItem):
        return PolygonObjectDetectionEvalItem(item, shape_key)
    return item


def iou_bboxes(item_gt, item_pred, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.total_iou(item_gt, label_weights, per_label=per_label)


def iou_bboxes_textarea(item_gt, item_pred, label_weights=None, shape_key=None,
                        algorithm='Levenshtein', qval=1, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.total_iou(item_gt, label_weights, algorithm=algorithm, qval=qval, per_label=per_label)


def iou_polygons(item_gt, item_pred, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_polygons(item_gt, shape_key=shape_key)
    item_pred = _as_polygons(item_pred, shape_key=shape_key)
    return item_pred.total_iou(item_gt, label_weights, per_label=per_label)


def iou_polygons_textarea(item_gt, item_pred, label_weights=None, shape_key=None,
                          algorithm='Levenshtein', qval=1, per_label=False):
    item_gt = _as_polygons(item_gt, shape_key=shape_key)
    item_pred = _as_polygons(item_pred, shape_key=shape_key)
    return item_pred.total_iou(item_gt, label_weights, algorithm=algorithm, qval=qval, per_label=per_label)


def precision_bboxes(item_gt, item_pred, iou_threshold=0.5, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.precision_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def precision_bboxes_for_map(item_gt, item_pred, iou_threshold=0.5, label_weights=None,
                             shape_key=None, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.precision_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def precision_polygons(item_gt, item_pred, iou_threshold=0.5, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_polygons(item_gt, shape_key=shape_key)
    item_pred = _as_polygons(item_pred, shape_key=shape_key)
    return item_pred.precision_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def recall_bboxes(item_gt, item_pred, iou_threshold=0.5, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.recall_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def recall_polygons(item_gt, item_pred, iou_threshold=0.5, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_polygons(item_gt, shape_key=shape_key)
    item_pred = _as_polygons(item_pred, shape_key=shape_key)
    return item_pred.recall_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def f1_bboxes(item_gt, item_pred, iou_threshold=0.5, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.f1_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def f1_polygons(item_gt, item_pred, iou_threshold=0.5, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_polygons(item_gt, shape_key=shape_key)
    item_pred = _as_polygons(item_pred, shape_key=shape_key)
    return item_pred.f1_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def mAP_bboxes(item_gt, item_pred, iou_threshold=0.5, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.mAP_at_iou(item_gt, iou_threshold, label_weights, per_label=per_label)


def prediction_bboxes(item_gt, item_pred, iou_threshold=0.5, shape_key=None):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_pred.prediction_result_at_iou_for_all_bbox(item_gt, iou_threshold)


def matrix_iou_bboxes(item_gt, item_pred, label_weights=None, shape_key=None, per_label=False):
    item_gt = _as_bboxes(item_gt, shape_key=shape_key)
    item_pred = _as_bboxes(item_pred, shape_key=shape_key)
    return item_gt.total_iou_matrix(item_pred, label_weights, per_label=per_label)
