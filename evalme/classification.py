from collections import defaultdict

from evalme.eval_item import EvalItem

import logging
logger = logging.getLogger(__name__)


class ClassificationEvalItem(EvalItem):

    SHAPE_KEY = 'undefined'

    def exact_match(self, item, label_weights=None, per_label=False):
        label_weights = label_weights or {}
        if self.empty and item.empty:
            return {} if per_label else 1
        if self.empty ^ item.empty and not per_label:
            return 0
        n = 0
        if per_label:
            total_weight = defaultdict(int)
        else:
            total_weight = 0
        for x, y in zip(self.get_values_iter(), item.get_values_iter()):
            try:
                labels = x[self._shape_key]
                y_labels = y[self._shape_key]
            except KeyError as exc:
                logger.error(f'Shapes of compared items are different!'
                             f'Reason: {exc}', exc_info=True)
                # different types of results:
                if per_label:
                    return {'Error': 0}
                else:
                    return 0
            if not isinstance(labels, list):
                labels = [labels]

            if not isinstance(y_labels, list):
                y_labels = [y_labels]
            # choices are mismatched
            if labels != y_labels:
                if per_label:
                    for l in labels:
                        total_weight[l] = 0
                else:
                    return 0
            # choices are matched
            else:
                if per_label:
                    # per label mode: label weights are unimportant
                    for l in labels:
                        total_weight[l] = 1
                else:
                    # aggregation mode: average scores by label weights
                    weight = sum(label_weights.get(l, 1) for l in labels)
                    total_weight += weight
                    n += len(labels)
        if per_label:
            return total_weight
        if n == 0:
            return 0
        return total_weight / n


class ChoicesEvalItem(ClassificationEvalItem):
    SHAPE_KEY = 'choices'


class PairwiseEvalItem(ClassificationEvalItem):
    SHAPE_KEY = 'pairwise'


def _as_choices(item, shape_key):
    if not isinstance(item, ChoicesEvalItem):
        return ChoicesEvalItem(item, shape_key=shape_key)
    return item


def _as_pairwise(item, shape_key):
    if not isinstance(item, PairwiseEvalItem):
        return PairwiseEvalItem(item, shape_key=shape_key)
    return item


def exact_matching_choices(item_gt, item_pred, label_weights=None, per_label=False, shape_key=None):
    return _as_choices(item_gt, shape_key).exact_match(_as_choices(item_pred, shape_key), label_weights, per_label=per_label)


def exact_matching_pairwise(item_gt, item_pred, label_weights=None, per_label=False, shape_key=None):
    return _as_pairwise(item_gt, shape_key).exact_match(_as_pairwise(item_pred, shape_key), label_weights, per_label=per_label)


def naive(x, y, per_label=False, **kwargs):
    """
    Naive comparison of annotations
    """
    if len(x) != len(y):
        result = 0
    else:
        for i in range(len(x)):
            if x[i]['value'] != y[i]['value']:
                result = 0
                break
        else:
            result = 1
    if per_label:
        return {'No-label': result}
    else:
        return result