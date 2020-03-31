import numpy as np

from functools import partial

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize

from evalme.eval_item import EvalItem


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
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def total_iou(self, item, label_weights=None):
        """
        For each shape in current eval item, we compute IOU with identically labeled shape with largest intersection.
        This is suboptimal metric since it doesn't consider cases where multiple boxes from self coincides with
        with single box from item
        :param item to be compared with self:
        :return:
        """
        ious = []
        label_weights = label_weights or {}
        shapes_pred = item.get_values()
        weights = []
        for shape in self.get_values_iter():
            shape_labels = shape[self._shape_key]
            weight = sum(label_weights.get(l, 1) for l in shape_labels)
            pred_bboxes_by_label = filter(lambda b: b[self._shape_key] == shape_labels, shapes_pred)
            iou_per_box = map(partial(self._iou, shape), pred_bboxes_by_label)
            try:
                iou = max(iou_per_box)
            except ValueError:
                iou = 0.0
            ious.append(weight * iou)
            weights.append(weight)
        return np.average(ious, weights=weights) if ious else 0.0

    def _precision_recall_at_iou(self, item, iou_threshold, label_weights=None):
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

    def precision_at_iou(self, item, iou_threshold=0.5, label_weights=None):
        precision, _ = self._precision_recall_at_iou(item, iou_threshold, label_weights)
        return precision

    def recall_at_iou(self, item, iou_threshold=0.5, label_weights=None):
        _, recall = self._precision_recall_at_iou(item, iou_threshold, label_weights)
        return recall

    def f1_at_iou(self, item, iou_threshold=0.5, label_weights=None):
        precision, recall = self._precision_recall_at_iou(item, iou_threshold, label_weights)
        return 2 * precision * recall / (precision + recall)


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

        # Polygon contains small bowties, we can fix them (https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera)
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


def _as_bboxes(item):
    if not isinstance(item, BboxObjectDetectionEvalItem):
        return BboxObjectDetectionEvalItem(item)
    return item


def _as_polygons(item):
    if not isinstance(item, PolygonObjectDetectionEvalItem):
        return PolygonObjectDetectionEvalItem(item)
    return item


def iou_bboxes(item_gt, item_pred, label_weights=None):
    return _as_bboxes(item_pred).total_iou(_as_bboxes(item_gt), label_weights)


def iou_polygons(item_gt, item_pred, label_weights=None):
    return _as_polygons(item_pred).total_iou(_as_polygons(item_gt), label_weights)


def precision_bboxes(item_gt, item_pred, iou_threshold=0.5, label_weights=None):
    return _as_bboxes(item_pred).precision_at_iou(_as_bboxes(item_gt), iou_threshold, label_weights)


def precision_polygons(item_gt, item_pred, iou_threshold=0.5, label_weights=None):
    return _as_polygons(item_pred).precision_at_iou(_as_polygons(item_gt), iou_threshold, label_weights)


def recall_bboxes(item_gt, item_pred, iou_threshold=0.5, label_weights=None):
    return _as_bboxes(item_pred).recall_at_iou(_as_bboxes(item_gt), iou_threshold, label_weights)


def recall_polygons(item_gt, item_pred, iou_threshold=0.5, label_weights=None):
    return _as_polygons(item_pred).recall_at_iou(_as_polygons(item_gt), iou_threshold, label_weights)


def f1_bboxes(item_gt, item_pred, iou_threshold=0.5, label_weights=None):
    return _as_bboxes(item_pred).f1_at_iou(_as_bboxes(item_gt), iou_threshold, label_weights)


def f1_polygons(item_gt, item_pred, iou_threshold=0.5, label_weights=None):
    return _as_polygons(item_pred).f1_at_iou(_as_polygons(item_gt), iou_threshold, label_weights)
