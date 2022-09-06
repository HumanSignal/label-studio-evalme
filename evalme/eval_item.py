from operator import itemgetter
from functools import partial

class EvalItem(object):
    """
    Generic class that contains all info about evaluation item
    """

    SHAPE_KEY = None

    def __init__(self, raw_data, shape_key=None, **kwargs):
        self._raw_data = raw_data
        self._shape_key = shape_key or self.SHAPE_KEY
        self._kwargs = kwargs
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
    def has_spans(results_list):
        for r in results_list:
            if not 'start' in r or not 'end' in r:
                return False
        return True

    @staticmethod
    def spans_iou(x, y):
        """
        Intersection over union for spans with start/end
        :param x:
        :param y:
        :return:
        """
        assert EvalItem.has_spans([x, y])
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
    def bbox_iou(boxA, boxB):
        """
        Intersection over union for bbox with x/y/width/height
        :param boxA:
        :param boxB:
        :return:
        """
        # check data
        assert boxA['x']
        assert boxB['x']
        assert boxA['y']
        assert boxB['y']
        assert boxA['width']
        assert boxB['width']
        assert boxA['height']
        assert boxB['height']

        # identify max coordinates
        xA = max(boxA['x'], boxB['x'])
        yA = max(boxA['y'], boxB['y'])
        # identify min coordinates
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

    @staticmethod
    def spans_iou_by_start_end_offsets(x, y):
        assert x['start']
        assert x['end']
        assert y['start']
        assert y['end']
        assert x['startOffset']
        assert x['endOffset']
        assert y['startOffset']
        assert y['endOffset']

        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s1 != s2 or e1 != e2:
            return 0

        s1, e1 = x['startOffset'], x['endOffset']
        s2, e2 = y['startOffset'], y['endOffset']
        if s2 > e1 or s1 > e2:
            return 0

        intersection = min(e1, e2) - max(s1, s2)
        union = max(e1, e2) - min(s1, s2)
        if union == 0:
            return 0
        iou = intersection / union
        return iou

    def max_score(self, prediction, matcher=None, check_condition=False):
        assert matcher
        gt = self.get_values_iter()
        pred = prediction.get_values_iter()
        max_score = 0
        for gt_value in gt:
            if check_condition:
                best_matching_score = max(map(partial(matcher, gt_value, check_condition=check_condition), pred),
                                          key=lambda r: r[1])
                if best_matching_score[1] == 0:
                    best_matching_score = 0
                else:
                    best_matching_score = best_matching_score[0]
            else:
                best_matching_score = max(map(partial(matcher, gt_value), pred))
            max_score = max(max_score, best_matching_score)
        return max_score

    def min_score(self, prediction, matcher=None):
        assert matcher
        gt = self.get_values_iter()
        pred = prediction.get_values_iter()
        max_score = 1
        for g in gt:
            for p in pred:
                max_score = min(max_score, matcher(g, p))
        return max_score

    def average_score(self, prediction, matcher=None):
        assert matcher
        gt = self.get_values_iter()
        pred = prediction.get_values_iter()
        max_score = 0
        n = 0
        for g in gt:
            for p in pred:
                max_score += matcher(g, p)
                n += 1
        return max_score / max(n, 1)
