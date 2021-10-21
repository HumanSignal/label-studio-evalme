import itertools
import re
import random

import numpy
import numpy as np

from collections import defaultdict, Counter

from label_studio_converter.brush import decode_rle
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize

from evalme.eval_item import EvalItem
from evalme.utils import get_text_comparator, texts_similarity, Result
from shapely.validation import explain_validity

from evalme.text.text import TextAreaEvalItem


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

    def _area_close(self, p1, p2, distance=0.1):
        return abs(p1.area / max(p2.area, 1e-8) - 1) < distance

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
        # trying to fix polygon with dilation
        distance = 0.01
        while distance < 11:
            fixed_poly = poly.buffer(distance)
            flag = self._area_close(fixed_poly, poly)
            if fixed_poly.is_valid and flag:
                return fixed_poly
            else:
                distance = distance * 2

        # trying to fix polygon with errosion
        distance = -0.01
        while distance > -11:
            fixed_poly = poly.buffer(distance)
            flag = self._area_close(fixed_poly, poly)
            if fixed_poly.is_valid and flag:
                return fixed_poly
            else:
                distance = distance * 2

        # trying to delete points near loop
        poly1 = self._remove_points(poly, points)
        if poly1.is_valid and self._area_close(poly1, poly):
            return poly1

        # We are failing to build polygon, this shall be reported via error log
        raise ValueError(f'Fail to build polygon from {points}')

    def _remove_points(self, poly, points):
        """
        Trying to remove some points to make polygon valid
        """
        removed_points = []
        invalidity = explain_validity(poly)
        for i in range(len(points)-4):
            match = re.findall("\d+\.\d+", invalidity)
            min_distance = poly.area
            min_point = None
            if match:
                intersect_point = (float(match[0]), float(match[1]))
                for point in points:
                    distance = abs(point[0] - intersect_point[0]) + abs(point[1] - intersect_point[1])
                    if distance < min_distance:
                        min_distance = distance
                        min_point = point
                removed_points.append(min_point)
                points.remove(min_point)
                poly1 = Polygon(points)
            if poly1.is_valid:
                break
            invalidity = explain_validity(poly1)
        if self._area_close(poly, poly1):
            return poly1
        random.shuffle(removed_points)
        for item in removed_points:
            points.append(item)
            poly2 = Polygon(points)
            poly2 = poly2.buffer(0)
            if poly2.is_valid:
                continue
            else:
                points.remove(item)
        return poly2

    def _iou(self, polyA, polyB):
        if polyA.get('points') and polyB.get('points'):
            pA = self._try_build_poly(polyA['points'])
            pB = self._try_build_poly(polyB['points'])
            inter_area = pA.intersection(pB).area
            iou = inter_area / (pA.area + pB.area - inter_area)
        else:
            iou = 0
        return iou


class KeyPointsEvalItem(EvalItem):
    SHAPE_KEY = 'keypoints'

    def equals(self, pred):
        # pred - predicted value
        results = [0] * len(self._raw_data)
        for i in range(len(self._raw_data)):
            result = 0
            gt_item = self._raw_data[i]
            for item in pred._raw_data:
                if item.x == gt_item.x and item.y == gt_item.y:
                    result = 1
            results[i] = result
        return sum(results) / len(results)

    def distance(self, pred, local_range=1, label_weights=None, per_label=False):
        if per_label:
            results = defaultdict(float)
            counts = defaultdict(int)
        else:
            results = 0
        for i in range(len(self._raw_data)):
            weight = 0
            gt_item = self._raw_data[i]
            gt_x = float(gt_item['value']['x'])
            gt_y = float(gt_item['value']['y'])
            gt_label = gt_item['value']['keypointlabels'][0]
            for item in pred._raw_data:
                pred_x = float(item['value']['x'])
                pred_y = float(item['value']['y'])
                pred_label = item['value']['keypointlabels'][0]
                if (abs(pred_x - gt_x) < local_range) and (abs(pred_y - gt_y) < local_range) and (pred_label == gt_label):
                    weight = label_weights[gt_label] if label_weights.get(gt_label) else 1
                    break
            if per_label:
                results[gt_label] += weight
                counts[gt_label] += 1
            else:
                results += weight

        if per_label:
            final_results = dict()
            for key in counts:
                final_results[key] = results[key] / counts[key]
        else:
            final_results = results / len(self._raw_data)

        return final_results


class OCREvalItem(ObjectDetectionEvalItem):
    SHAPE_KEY = 'rectangle'

    def compare(self, pred, threshold=0.5, algorithm='Levenshtein', per_label=False):
        if per_label:
            results = defaultdict(float)
            num_results = defaultdict(int)
        else:
            results = dict()

        gt_ids = self._get_ids_from_results()
        pred_ids = pred._get_ids_from_results()

        for id_gt in gt_ids:
            gt_results = self._get_results_by_id(id_gt)
            gt_types = self._get_types_from_results(gt_results)
            for id_pred in pred_ids:
                pred_results = pred._get_results_by_id(id_pred)
                pred_types = self._get_types_from_results(pred_results)
                if 'rectangle' in pred_types and 'rectangle' in gt_types:
                    gt_results_rectangle = [item for item in gt_results if item['type'] == 'rectangle']
                    pred_results_rectangle = [item for item in pred_results if item['type'] == 'rectangle']
                    score = self._get_max_iou_rectangles(gt_results_rectangle, pred_results_rectangle, threshold)
                    if score < threshold:
                        results[id_gt] = 0
                    else:
                        gt_results_labels = [item['value']['labels'] for item in gt_results if item['type'] == 'labels']
                        pred_results_labels = [item['value']['labels'] for item in pred_results if item['type'] == 'labels']
                        if gt_results_labels == pred_results_labels:
                            gt_results_text = TextAreaEvalItem([item for item in gt_results if item['type'] != 'labels' and item['type'] != 'rectangle'])
                            pred_results_text = TextAreaEvalItem([item for item in pred_results if item['type'] != 'labels' and item['type'] != 'rectangle'])
                            res = gt_results_text.match(item=pred_results_text, algorithm=algorithm)
                            if per_label:
                                for item in pred_results_labels:
                                    for subitem in item:
                                        results[subitem] += res
                                        num_results[subitem] += 1
                            else:
                                results[id_gt] = res
                        else:
                            results[id_gt] = 0
                else:
                    continue

        if per_label:
            final_results = {}
            for item in results:
                final_results[item] = results[item] / max(num_results[item], 1)
            return final_results
        else:
            values = results.values()
            return sum(values) / len(values) if len(values) > 0 else 0


    def _get_max_iou_rectangles(self, gt, pred, threshold):
        max_score = 0
        for item in gt:
            for pred_item in pred:
                score = self._iou(item['value'], pred_item['value'])
                max_score = max(max_score, score)
        return max_score

    def _get_types_from_results(self, results):
        """
        Get types from results
        """
        res = set()
        for result in results:
            r = result.get('type')
            if r:
                res.add(r)
        return res

    def _get_results_by_id(self, id):
        """
        Get results by ID
        """
        res = []
        for result in self._raw_data:
            if result.get('id') == id:
                res.append(result)
        return res

    def _get_ids_from_results(self):
        """
        Get IDs from results to group
        """
        res = set()
        for result in self._raw_data:
            id = result.get('id')
            if id:
                res.add(id)
        return list(res)


class BrushEvalItem(ObjectDetectionEvalItem):
    SHAPE_KEY = 'brushlabels'

    @staticmethod
    def _iou(gt, pred):
        gt = decode_rle(gt['rle'])
        pred = decode_rle(pred['rle'])
        union = np.logical_or(gt, pred).sum()
        intersection = np.logical_and(gt, pred).sum()
        return intersection / max(union, 1)

    def iou(self, pred_item, per_label=False, label_weights=None):
        if per_label:
            ious = defaultdict(list)
        else:
            ious, weights = [], []
        for gt in self.get_values_iter():
            max_iou = 0
            for pred in pred_item.get_values_iter():
                if not gt[self._shape_key] == pred[self._shape_key]:
                    continue
                iou = self._iou(gt, pred)
                max_iou = max(iou, max_iou)
            if per_label:
                for l in gt[self._shape_key]:
                    ious[l].append(max_iou)
            else:
                weight = sum(label_weights.get(l, 1) for l in gt[self._shape_key]) / max(len(gt[self._shape_key]), 1)
                ious.append(max_iou * weight)
                weights.append(weight)
        if per_label:
            return {l: float(np.mean(v)) for l, v in ious.items()}
        return np.average(ious, weights=weights) if ious else 0.0


def _as_bboxes(item, shape_key=None):
    if not isinstance(item, BboxObjectDetectionEvalItem):
        return BboxObjectDetectionEvalItem(item, shape_key)
    return item


def _as_polygons(item, shape_key=None):
    if not isinstance(item, PolygonObjectDetectionEvalItem):
        return PolygonObjectDetectionEvalItem(item, shape_key)
    return item


def _as_keypoint(item):
    if not isinstance(item, KeyPointsEvalItem):
        return KeyPointsEvalItem(item)
    return item


def _as_ocreval(item):
    if not isinstance(item, OCREvalItem):
        return OCREvalItem(item)
    return item


def _as_brush(item):
    if not isinstance(item, BrushEvalItem):
        return BrushEvalItem(item)
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


def keypoints_distance(item_gt, item_pred, per_label=False, label_weights=None):
    item_gt = _as_keypoint(item_gt)
    item_pred = _as_keypoint(item_pred)
    return item_gt.distance(item_pred, label_weights=label_weights, per_label=per_label)


def ocr_compare(item_gt, item_pred, per_label=False, iou_threshold=0.5, algorithm='Levenshtein', label_weights=None):
    item_gt = _as_ocreval(item_gt)
    item_pred = _as_ocreval(item_pred)
    return item_gt.compare(item_pred, per_label=per_label, threshold=iou_threshold, algorithm=algorithm)


def iou_brush(item_gt, item_pred, per_label=False, label_weights=None):
    item_gt = _as_brush(item_gt)
    item_pred = _as_brush(item_pred)
    return item_gt.iou(item_pred, per_label=per_label, label_weights=label_weights)


def precision_brush(item_gt, item_pred, iou_threshold=0.5, label_weights=None, per_label=False):
    item_gt = _as_brush(item_gt)
    item_pred = _as_brush(item_pred)
    return item_gt.precision_at_iou(item_pred, iou_threshold, label_weights, per_label=per_label)


def recall_brush(item_gt, item_pred, iou_threshold=0.5, label_weights=None, per_label=False):
    item_gt = _as_brush(item_gt)
    item_pred = _as_brush(item_pred)
    return item_gt.recall_at_iou(item_pred, iou_threshold, label_weights, per_label=per_label)


def f1_brush(item_gt, item_pred, iou_threshold=0.5, label_weights=None, per_label=False):
    item_gt = _as_brush(item_gt)
    item_pred = _as_brush(item_pred)
    return item_gt.f1_at_iou(item_pred, iou_threshold, label_weights, per_label=per_label)
