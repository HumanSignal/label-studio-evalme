from functools import partial

from evalme.eval_item import EvalItem
from evalme.utils import texts_similarity, get_text_comparator


class TextTagsEvalItem(EvalItem):

    SHAPE_KEY = 'labels'

    def spans_iou(self, x, y):
        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s2 > e1 or s1 > e2:
            return 0
        intersection = min(e1, e2) - max(s1, s2)
        union = max(e1, e2) - min(s1, s2)
        if union == 0:
            return 0
        iou = intersection / union
        return iou

    def _match(self, x, y, f):
        labels_match = texts_similarity(x[self._shape_key], y[self._shape_key], f)
        spans_match = self.spans_iou(x, y)
        return labels_match * spans_match

    def intersection(self, item, label_weights=None, algorithm=None, qval=None):
        comparator = get_text_comparator(algorithm, qval)
        label_weights = label_weights or {}
        someone_is_empty = self.empty ^ item.empty
        if someone_is_empty:
            return 0
        if self.empty and item.empty:
            return 1

        gt_values = self.get_values()
        total_score, total_weight = 0, 0
        for pred_value in item.get_values_iter():
            # find the best matching span inside gt_values
            best_matching_score = max(map(partial(self._match, y=pred_value, f=comparator), gt_values))

            if self._shape_key in pred_value:
                weight = sum(label_weights.get(l, 1) for l in pred_value[self._shape_key])
            else:
                weight = 1
            total_score += weight * best_matching_score
            total_weight += weight
        if total_weight == 0:
            return 0
        return total_score / total_weight


class HTMLTagsEvalItem(TextTagsEvalItem):
    SHAPE_KEY = 'htmllabels'

    def spans_iou(self, x, y):
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


class TextAreaEvalItem(EvalItem):
    SHAPE_KEY = 'textarea'

    def match(self, item, algorithm='Levenshtein', qval=1):
        comparator = get_text_comparator(algorithm, qval)
        all_scores = []
        for gt, pred in zip(self.get_values_iter(), item.get_values_iter()):
            all_scores.append(texts_similarity(gt, pred, comparator))
        return sum(all_scores) / max(len(all_scores), 1)


def _as_text_tags_eval_item(item, shape_key):
    if not isinstance(item, TextTagsEvalItem):
        return TextTagsEvalItem(item, shape_key=shape_key)
    return item


def _as_html_tags_eval_item(item, shape_key):
    if not isinstance(item, HTMLTagsEvalItem):
        return HTMLTagsEvalItem(item, shape_key=shape_key)
    return item


def _as_textarea_eval_item(item):
    if not isinstance(item, HTMLTagsEvalItem):
        return TextAreaEvalItem(item)
    return item


def intersection_text_tagging(item_gt, item_pred, label_weights=None, shape_key=None):
    item_gt = _as_text_tags_eval_item(item_gt, shape_key=shape_key)
    item_pred = _as_text_tags_eval_item(item_pred, shape_key=shape_key)
    return item_gt.intersection(item_pred, label_weights)


def intersection_textarea_tagging(item_gt, item_pred, label_weights=None, shape_key='text', algorithm='Levenshtein', qval=1):
    item_gt = _as_text_tags_eval_item(item_gt, shape_key=shape_key)
    item_pred = _as_text_tags_eval_item(item_pred, shape_key=shape_key)
    return item_gt.intersection(item_pred, label_weights=label_weights, algorithm=algorithm, qval=qval)


def intersection_html_tagging(item_gt, item_pred, label_weights=None, shape_key=None, algorithm=None, qval=None):
    item_gt = _as_html_tags_eval_item(item_gt, shape_key=shape_key)
    item_pred = _as_html_tags_eval_item(item_pred, shape_key=shape_key)
    return item_gt.intersection(item_pred, label_weights, algorithm=algorithm, qval=qval)


def match_textareas(item_gt, item_pred, algorithm='Levenshtein', qval=1):
    qval = int(qval or 0) or None
    item_gt = _as_textarea_eval_item(item_gt)
    item_pred = _as_textarea_eval_item(item_pred)
    return item_gt.match(item_pred, algorithm, qval)
