import itertools
import re
import random

import numpy
import numpy as np

from collections import defaultdict, Counter
from operator import itemgetter

from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
from shapely.validation import explain_validity
from shapely.validation import make_valid

from evalme.utils import get_text_comparator, texts_similarity, Result
from label_studio_converter.brush import decode_rle


class EvalItem(object):
    """
    Generic class that contains all info about evaluation item
    """

    SHAPE_KEY = None

    def __init__(self, raw_data, shape_key=None):
        """
        :param raw_data: Annotation result
        :param shape_key:
        """
        self._raw_data = raw_data
        self._shape_key = shape_key or self.SHAPE_KEY
        if not self._shape_key:
            raise ValueError('Shape key is undefined')

    @property
    def raw_data(self):
        return self._raw_data

    def get_values_iter(self):
        return map(itemgetter('value'), self._raw_data)

    def get_values(self):
        return list(self.get_values_iter())

    @property
    def empty(self):
        return len(self._raw_data) == 0

    def __len__(self):
        return len(self._raw_data)

    @staticmethod
    def _get_ids_from_results(_raw_data):
        """
        Get result IDs from results to group
        """
        res = defaultdict(lambda: defaultdict(list))
        for result in _raw_data['result']:
            id = result.get('id')
            res[id][result['type'].lower()].append(result)
        return res

    @staticmethod
    def _get_types_from_results(results: list):
        """
        Get types from results
        :param results: Annotation results List
        :return: set of Results types
        """
        res = set()
        for result in results:
            r = result.get('type')
            if r:
                res.add(r)
        return res

    @staticmethod
    def _get_max_iou_rectangles(_iou, gt, pred):
        """
        Get max iou for OCR shapes
        :param _iou: Comparing function
        :param gt: Ground Truth result
        :param pred: Predicted result
        :return: Max score float[0..1]
        """
        max_score = 0
        for item in gt:
            for pred_item in pred:
                score = _iou(item['value'], pred_item['value'])
                max_score = max(max_score, score)
        return max_score

    @staticmethod
    def _per_region_result(result):
        """
        Check if result is per_region
        :param result: Json with result
        :return: True if result is per_region
        """
        if ('start' in result['value'] and 'end' in result['value']) or (
                'x' in result['value'] and 'y' in result['value']) or ('points' in result['value']):
            return True
        return False

    @staticmethod
    def spans_iou(x, y):
        s1, e1 = float(x['start']), float(x['end'])
        s2, e2 = float(y['start']), float(y['end'])
        if s2 > e1 or s1 > e2:
            return 0
        intersection = min(e1, e2) - max(s1, s2)
        union = max(e1, e2) - min(s1, s2)
        if union == 0:
            return 0
        iou = intersection / union
        return iou

    @staticmethod
    def spans_iou_by_start_end_offsets(x, y):
        """This code handles IOU for spans, but fails when start/end point to different blocks"""
        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s1 != s2 or e1 != e2:
            return 0

        s1, e1 = x['startOffset'], x['endOffset']
        s2, e2 = y['startOffset'], y['endOffset']
        if s2 > e1 or s1 > e2:
            return 0

        # correct end offset if they lie in different block
        # if e1 <= s1:
        #     e1 = s1 + len(x.get('text', ''))
        # if e2 <= s2:
        #     e2 = s2 + len(y.get('text', ''))

        intersection = min(e1, e2) - max(s1, s2)
        union = max(e1, e2) - min(s1, s2)
        if union == 0:
            return 0
        iou = intersection / union
        return iou

    @staticmethod
    def bbox_iou(boxA, boxB):
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

    # Polygon methods
    @staticmethod
    def poly_iou(polyA, polyB):
        if polyA.get('points') and polyB.get('points'):
            pA = _try_build_poly(polyA['points'])
            pB = _try_build_poly(polyB['points'])
            inter_area = pA.intersection(pB).area
            iou = inter_area / (pA.area + pB.area - inter_area)
        else:
            iou = 0
        return iou

    @staticmethod
    def _area_close(p1, p2, distance=0.1):
        return abs(p1.area / max(p2.area, 1e-8) - 1) < distance

    @staticmethod
    def _try_build_poly(points):
        poly = Polygon(points)
        # Everything is OK, points are valid polygon
        if poly.is_valid:
            return poly
        # try making valid polygon
        poly = make_valid(poly)
        if poly.is_valid:
            return poly

        # Polygon contains small bowties, we can fix them:
        # (https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera)
        fixed_poly = poly.buffer(0)
        if not fixed_poly.geom_type == 'MultiPolygon' and EvalItem._area_close(fixed_poly, poly):
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
        if EvalItem._area_close(convex_hull, poly):
            return convex_hull
        # trying to fix polygon with dilation
        distance = 0.01
        while distance < 11:
            fixed_poly = poly.buffer(distance)
            flag = EvalItem._area_close(fixed_poly, poly)
            if fixed_poly.is_valid and flag:
                return fixed_poly
            else:
                distance = distance * 2

        # trying to fix polygon with errosion
        distance = -0.01
        while distance > -11:
            fixed_poly = poly.buffer(distance)
            flag = EvalItem._area_close(fixed_poly, poly)
            if fixed_poly.is_valid and flag:
                return fixed_poly
            else:
                distance = distance * 2

        # trying to delete points near loop
        poly1 = EvalItem._remove_points(poly, points)
        if poly1.is_valid and EvalItem._area_close(poly1, poly):
            return poly1

        # We are failing to build polygon, this shall be reported via error log
        raise ValueError(f'Fail to build polygon from {points}')

    @staticmethod
    def _remove_points(poly, points):
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

    @staticmethod
    def get_best_matching_result(result, list_to_compare: List, compare_f, threshold=0):
        """
        Get best matching result by compare function
        :param result:
        :param list_to_compare:
        :param compare_f:
        :return:
        """
        max_score = 0
        final_result = None
        for item in list_to_compare:
            score = compare_f(result, item)
            if score > max_score:
                max_score = score
                final_result = item
        if threshold < max_score:
            return final_result
        else:
            return None

    @staticmethod
    def identify_region_comparing_function(result1, result2):
        """
        Identify region comparision function by keys in results
        :param result1:
        :param result2:
        :return: function or None
        """
        polygon = ['points']
        spans_with_offset = ['start', 'end', 'startOffset', 'endOffset']
        spans = ['start', 'end']
        bbox = ['x', 'y', 'width', 'height']
        if all(key in result1['value'] for key in polygon) and all(key in result2['value'] for key in polygon):
            return EvalItem.poly_iou
        elif all(key in result1['value'] for key in spans_with_offset) and all(key in result2['value'] for key in spans_with_offset):
            return EvalItem.spans_iou_by_start_end_offsets
        elif all(key in result1['value'] for key in spans) and all(key in result2['value'] for key in spans):
            return EvalItem.spans_iou
        elif all(key in result1['value'] for key in bbox) and all(key in result2['value'] for key in bbox):
            return EvalItem.bbox_iou
        else:
            return None
