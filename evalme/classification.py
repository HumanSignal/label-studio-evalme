from collections import defaultdict

from evalme.eval_item import EvalItem


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
            labels = x[self._shape_key]
            if not isinstance(labels, list):
                labels = [labels]
            # choices are mismatched
            if labels != y[self._shape_key]:
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
