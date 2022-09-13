import xmljson
import textdistance
import random
import re

from itertools import zip_longest
from collections import Counter
from enum import Enum
from lxml import etree

from shapely.validation import explain_validity
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize

_text_comparators = {}


class Result(Enum):
    UN = 0
    TP = 1
    FP = 2
    FN = 3
    TN = 4


def get_text_comparator(algorithm, qval):
    if algorithm is None and qval is None:
        # the default comparator fallback to simple text/labels equality
        return None
    comparator_key = (algorithm, qval)
    if comparator_key not in _text_comparators:
        _text_comparators[comparator_key] = getattr(textdistance, algorithm)(qval=qval)
    return _text_comparators[comparator_key].normalized_similarity


def texts_similarity(x, y, f=None):
    if not f:
        # the default comparator fallback to simple text/labels equality
        return x == y
    scores = []
    for xi, yi in zip_longest(x, y):
        if xi is None or yi is None:
            scores.append(0)
        else:
            scores.append(f(xi, yi))
    mean_score = sum(scores) / max(len(scores), 1)
    return mean_score


def calculate_ap(results):
    """
    Calculation AP for results list
    Method is used for bbox image detection
    :return: Average precision: float[0..1]
    """
    if len(results) < 0:
        return 0
    precision = []
    recall = []
    total_true = sum(1 for item in results if item == Result.TP or item == Result.FN)
    if total_true < 1:
        return 0
    total_positive = 0
    tp = 0
    for item in results:
        if item == Result.TP:
            tp += 1
            total_positive += 1
        if item == Result.FP:
            total_positive += 1
        if total_positive > 0:
            precision.append(tp / total_positive)
        else:
            precision.append(0.0)
        recall.append(tp / total_true)
    for i in range(len(precision) - 1):
        m = max(precision[i:])
        precision[i] = max(m, precision[i])
    c = Counter(recall)
    c = sorted(c.items())
    start = 0
    res = {}
    for key in c:
        res[key[0]] = max(precision[start:start + key[1]])
        start += key[1]
    s = 0
    for i in range(0, 11):
        inc = i / 10
        for key in res:
            if inc > key:
                continue
            else:
                s += res[key]
                break
        else:
            s += res[key]
    return s / 11


def parse_config_to_json(config_string):
    parser = etree.XMLParser(recover=False)
    xml = etree.fromstring(config_string, parser)
    if xml is None:
        raise etree.XMLSchemaParseError('xml is empty or incorrect')
    config = xmljson.badgerfish.data(xml)
    return config


# POLYGON methods
def _area_close(p1, p2, distance=0.1):
    return abs(p1.area / max(p2.area, 1e-8) - 1) < distance


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
    if not fixed_poly.geom_type == 'MultiPolygon' and _area_close(fixed_poly, poly):
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
    if _area_close(convex_hull, poly):
        return convex_hull
    # trying to fix polygon with dilation
    distance = 0.01
    while distance < 11:
        fixed_poly = poly.buffer(distance)
        flag = _area_close(fixed_poly, poly)
        if fixed_poly.is_valid and flag:
            return fixed_poly
        else:
            distance = distance * 2

    # trying to fix polygon with errosion
    distance = -0.01
    while distance > -11:
        fixed_poly = poly.buffer(distance)
        flag = _area_close(fixed_poly, poly)
        if fixed_poly.is_valid and flag:
            return fixed_poly
        else:
            distance = distance * 2

    # trying to delete points near loop
    poly1 = _remove_points(poly, points)
    if poly1.is_valid and _area_close(poly1, poly):
        return poly1

    # We are failing to build polygon, this shall be reported via error log
    raise ValueError(f'Fail to build polygon from {points}')


def _remove_points(poly, points):
    """
    Trying to remove some points to make polygon valid
    """
    removed_points = []
    invalidity = explain_validity(poly)
    for i in range(len(points) - 4):
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
    if _area_close(poly, poly1):
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

