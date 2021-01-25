from functools import partial
from collections import defaultdict

from evalme.eval_item import EvalItem
from evalme.utils import texts_similarity, get_text_comparator


class TextTagsEvalItem(EvalItem):

    SHAPE_KEY = 'labels'

    def spans_iou(self, x, y):
        s1, e1 = int(x['start']), int(x['end'])
        s2, e2 = int(y['start']), int(y['end'])
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

    def intersection(self, item, label_weights=None, algorithm=None, qval=None, per_label=False, iou_threshold=None):
        comparator = get_text_comparator(algorithm, qval)
        label_weights = label_weights or {}
        someone_is_empty = self.empty ^ item.empty
        if someone_is_empty and not per_label:
            return 0
        if self.empty and item.empty:
            return {} if per_label else 1

        gt_values = self.get_values()
        if per_label:
            total_score, total_weight = defaultdict(int), defaultdict(int)
        else:
            total_score, total_weight = 0, 0
        for pred_value in item.get_values_iter():
            if len(gt_values) == 0:
                # for empty gt values, matching score for current prediction is the lowest
                best_matching_score = 0
            else:
                # find the best matching span inside gt_values
                best_matching_score = max(map(partial(self._match, y=pred_value, f=comparator), gt_values))
                if iou_threshold is not None:
                    # make hard decision w.r.t. threshold whether current spans are matched
                    best_matching_score = float(best_matching_score > iou_threshold)
            if per_label:
                # for per-label mode, label weights are unimportant - only scores are averaged
                for l in pred_value[self._shape_key]:
                    total_score[l] += best_matching_score
                    total_weight[l] += 1
            else:
                # when aggregating scores each individual label weight is taken into account
                if self._shape_key in pred_value:
                    weight = sum(label_weights.get(l, 1) for l in pred_value[self._shape_key])
                else:
                    weight = 1
                total_score += weight * best_matching_score
                total_weight += weight

        if per_label:
            # average per-label score
            for l in total_score:
                if total_weight[l] == 0:
                    total_score[l] = 0
                else:
                    total_score[l] /= total_weight[l]
            return total_score

        # otherwise return overall score
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


class TextAreaEvalItem(EvalItem):
    SHAPE_KEY = 'text'

    def match(self, item, algorithm='Levenshtein', qval=1):
        comparator = get_text_comparator(algorithm, qval)
        all_scores = []
        for gt, pred in zip(self.get_values_iter(), item.get_values_iter()):
            all_scores.append(texts_similarity(gt[self._shape_key], pred[self._shape_key], comparator))
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


def intersection_text_tagging(item_gt, item_pred, label_weights=None, shape_key=None, per_label=False, iou_threshold=None):
    item_gt = _as_text_tags_eval_item(item_gt, shape_key=shape_key)
    item_pred = _as_text_tags_eval_item(item_pred, shape_key=shape_key)
    return item_gt.intersection(item_pred, label_weights, per_label=per_label, iou_threshold=iou_threshold)


def intersection_textarea_tagging(item_gt, item_pred, label_weights=None, shape_key='text', algorithm='Levenshtein', qval=1, per_label=False, iou_threshold=None):
    item_gt = _as_text_tags_eval_item(item_gt, shape_key=shape_key)
    item_pred = _as_text_tags_eval_item(item_pred, shape_key=shape_key)
    return item_gt.intersection(item_pred, label_weights=label_weights, algorithm=algorithm, qval=qval, per_label=per_label, iou_threshold=iou_threshold)


def intersection_html_tagging(item_gt, item_pred, label_weights=None, shape_key=None, algorithm=None, qval=None, per_label=False, iou_threshold=None):
    item_gt = _as_html_tags_eval_item(item_gt, shape_key=shape_key)
    item_pred = _as_html_tags_eval_item(item_pred, shape_key=shape_key)
    return item_gt.intersection(item_pred, label_weights, algorithm=algorithm, qval=qval, per_label=per_label, iou_threshold=iou_threshold)


def match_textareas(item_gt, item_pred, algorithm='Levenshtein', qval=1, **kwargs):
    qval = int(qval or 0) or None
    item_gt = _as_textarea_eval_item(item_gt)
    item_pred = _as_textarea_eval_item(item_pred)
    if kwargs.get('per_label'):
        # per-label mode is not supported for the plain text area
        return {}
    return item_gt.match(item_pred, algorithm, qval)
