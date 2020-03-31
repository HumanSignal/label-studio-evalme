import textdistance

from functools import partial
from itertools import zip_longest

from evalme.eval_item import EvalItem


class TextTagsEvalItem(EvalItem):

    SHAPE_KEY = 'labels'

    def _intersect(self, x, y):
        if x[self._shape_key] != y[self._shape_key]:
            return 0
        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s2 > e1 or s1 > e2:
            return 0
        return min(e1, e2) - max(s1, s2)

    def intersection(self, item, label_weights=None):
        label_weights = label_weights or {}
        someone_is_empty = self.empty ^ item.empty
        if someone_is_empty:
            return 0
        if self.empty and item.empty:
            return 1
        total_len, total_intersect_len = 0, 0
        gt_values = self.get_values()
        for pred_value in item.get_values_iter():
            intersect_length = max(map(partial(self._intersect, y=pred_value), gt_values))
            length = pred_value['end'] - pred_value['start']
            if self._shape_key in pred_value:
                weight = sum(label_weights.get(l, 1) for l in pred_value[self._shape_key])
            else:
                weight = 1
            total_len += weight * length
            total_intersect_len += weight * intersect_length
        if total_len == 0:
            return 0
        return total_intersect_len / total_len


class HTMLTagsEvalItem(EvalItem):
    SHAPE_KEY = 'htmllabels'

    def _intersect(self, x, y):
        if x[self._shape_key] != y[self._shape_key]:
            return 0
        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s1 != s2 or e1 != e2:
            return 0
        s1, e1 = x['startOffset'], x['endOffset']
        s2, e2 = y['startOffset'], y['endOffset']
        if s2 > e1 or s1 > e2:
            return 0
        return min(e1, e2) - max(s1, s2)

    def intersection(self, item, label_weights=None):
        label_weights = label_weights or {}
        item = _as_html_tags_eval_item(item)
        someone_is_empty = self.empty ^ item.empty
        if someone_is_empty:
            return 0
        if self.empty and item.empty:
            return 1
        total_len, total_intersect_len = 0, 0
        gt_values = self.get_values()
        for pred_value in item.get_values_iter():
            intersect_length = max(map(partial(self._intersect, y=pred_value), gt_values))
            length = pred_value['endOffset'] - pred_value['startOffset']

            if self._shape_key in pred_value:
                weight = sum(label_weights.get(l, 1) for l in pred_value[self._shape_key])
            else:
                weight = 1

            total_len += weight * length
            total_intersect_len += weight * intersect_length
        if total_len == 0:
            return 0
        return total_intersect_len / total_len


class TextAreaEvalItem(EvalItem):
    SHAPE_KEY = 'textarea'

    _comparators = {}

    def _averaged_score(self, x, y, comparator):
        scores = []
        for gt_text, pred_text in zip_longest(x['text'], y['text']):
            if gt_text is None or pred_text is None:
                scores.append(0)
            else:
                scores.append(comparator(gt_text, pred_text))
        mean_score = sum(scores) / max(len(scores), 1)
        return mean_score

    def match(self, item, algorithm='Levenshtein', qval=1):
        comparator_key = (algorithm, qval)
        if comparator_key not in self._comparators:
            self._comparators[comparator_key] = getattr(textdistance, algorithm)(qval=qval)
        comparator = self._comparators[comparator_key].normalized_similarity
        all_scores = []
        for gt, pred in zip(self.get_values_iter(), item.get_values_iter()):
            all_scores.append(self._averaged_score(gt, pred, comparator))
        return sum(all_scores) / max(len(all_scores), 1)

    def _intersect(self, x, y, t, c):
        text_similarity = self._averaged_score(x, y, c)
        if text_similarity < t:
            return 0
        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s2 > e1 or s1 > e2:
            return 0
        return min(e1, e2) - max(s1, s2)

    def intersection(self, item, algorithm='Levenshtein', qval=1, threshold=0.5):
        comparator_key = (algorithm, qval)
        if comparator_key not in self._comparators:
            self._comparators[comparator_key] = getattr(textdistance, algorithm)(qval=qval)
        comparator = self._comparators[comparator_key].normalized_similarity
        someone_is_empty = self.empty ^ item.empty
        if someone_is_empty:
            return 0
        if self.empty and item.empty:
            return 1
        total_len, total_intersect_len = 0, 0
        gt_values = self.get_values()
        for pred_value in item.get_values_iter():
            intersect_length = max(map(partial(
                self._intersect, y=pred_value, t=threshold, c=comparator), gt_values))
            length = pred_value['end'] - pred_value['start']
            total_len += length
            total_intersect_len += intersect_length
        if total_len == 0:
            return 0
        return total_intersect_len / total_len


def _as_text_tags_eval_item(item, shape_key):
    if not isinstance(item, TextTagsEvalItem):
        return TextTagsEvalItem(item, shape_key=shape_key)
    return item


def _as_html_tags_eval_item(item):
    if not isinstance(item, HTMLTagsEvalItem):
        return HTMLTagsEvalItem(item)
    return item


def _as_textarea_eval_item(item):
    if not isinstance(item, HTMLTagsEvalItem):
        return TextAreaEvalItem(item)
    return item


def intersection_text_tagging(item_gt, item_pred, label_weights=None, shape_key=None):
    item_gt = _as_text_tags_eval_item(item_gt, shape_key=shape_key)
    item_pred = _as_text_tags_eval_item(item_pred, shape_key=shape_key)
    return item_gt.intersection(item_pred, label_weights)


def intersection_html_tagging(item_gt, item_pred, label_weights=None):
    return _as_html_tags_eval_item(item_gt).intersection(item_pred, label_weights)


def match_textareas(item_gt, item_pred, algorithm='Levenshtein', qval=1):
    qval = int(qval or 0) or None
    item_gt = _as_textarea_eval_item(item_gt)
    item_pred = _as_textarea_eval_item(item_pred)
    return item_gt.match(item_pred, algorithm, qval)


def intersection_textareas(item_gt, item_pred, algorithm='Levenshtein', qval=1, threshold=0.5):
    qval = int(qval or 0) or None
    item_gt = _as_textarea_eval_item(item_gt)
    item_pred = _as_textarea_eval_item(item_pred)
    return item_gt.intersection(item_pred, algorithm, qval, threshold=threshold)
