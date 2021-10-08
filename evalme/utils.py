from collections import Counter
from enum import Enum
from lxml import etree
import xmljson

import textdistance

from itertools import zip_longest

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