from collections import defaultdict

from evalme.eval_item import EvalItem

import logging
logger = logging.getLogger(__name__)


class ClassificationEvalItem(EvalItem):

    SHAPE_KEY = 'undefined'

    def exact_match(self, item, label_weights=None, per_label=False, label_order_matters=False):
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
                    return {}
                else:
                    return 0
            if not isinstance(labels, list):
                labels = [labels]
            if not isinstance(y_labels, list):
                y_labels = [y_labels]
            # Check if spans are mismatched
            mismatched_spans = False
            region = EvalItem.has_regions([x, y])
            if region:
                mismatched_spans = not bool(EvalItem.general_iou_by_type(region, x, y))
            # If order does not matter, sort labels
            if not label_order_matters:
                labels = sorted(labels)
                y_labels = sorted(y_labels)
            # choices are mismatched
            if labels != y_labels or mismatched_spans:
                if per_label:
                    for label in labels:
                        if isinstance(label, list):
                            for l in label:
                                total_weight[l] = 1 if label in y_labels else 0
                        else:
                            total_weight[label] = 0
                else:
                    return 0
            # choices are matched
            else:
                if per_label:
                    # per label mode: label weights are unimportant
                    for label in labels:
                        if isinstance(label, list):
                            for l in label:
                                total_weight[l] = 1
                        else:
                            total_weight[label] = 1
                else:
                    # aggregation mode: average scores by label weights
                    for label in labels:
                        if isinstance(label, list):
                            for l in label:
                                total_weight += label_weights.get(l, 1)
                                n += 1
                        else:
                            total_weight += label_weights.get(label, 1)
                            n += 1
        # if there are no results than 2nd result doesn't contain any result (null annotation)
        if not total_weight:
            # mark labels in 1st annotation with 0 score
            for x in self.get_values_iter():
                labels = x[self._shape_key]
                if not isinstance(labels, list):
                    labels = [labels]
                for label in labels:
                    total_weight[label] = 0
        if per_label:
            return total_weight
        if n == 0:
            return 0
        return total_weight / n


class ChoicesEvalItem(ClassificationEvalItem):
    SHAPE_KEY = 'choices'


class PairwiseEvalItem(ClassificationEvalItem):
    SHAPE_KEY = 'selected'


def _as_choices(item, shape_key, **kwargs):
    if not isinstance(item, ChoicesEvalItem):
        return ChoicesEvalItem(item, shape_key=shape_key, **kwargs)
    return item


def _as_pairwise(item, shape_key, **kwargs):
    if not isinstance(item, PairwiseEvalItem):
        return PairwiseEvalItem(item, shape_key=shape_key, **kwargs)
    return item


def exact_matching_choices(item_gt, item_pred, label_weights=None, per_label=False, shape_key=None, **kwargs):
    return _as_choices(item_gt, shape_key, **kwargs).exact_match(_as_choices(item_pred, shape_key, **kwargs),
                                                                 label_weights,
                                                                 per_label=per_label,
                                                                 label_order_matters=False)


def exact_matching_pairwise(item_gt, item_pred, label_weights=None, per_label=False, shape_key=None, **kwargs):
    return _as_pairwise(item_gt, shape_key, **kwargs).exact_match(_as_pairwise(item_pred, shape_key, **kwargs),
                                                                  label_weights,
                                                                  per_label=per_label)


def naive(x, y, per_label=False, label_order_matters=False, **kwargs):
    """
    Naive comparison of annotations

    If label order doesn't matter, we consider y's whole result array to find an exact match for each item from x['result'].
    This could be made more efficient by sorting the results first, but we don't do that yet.
    """
    # extract results from annotations
    if isinstance(x, dict) and isinstance(y, dict):
        x = x.get("result", [])
        y = y.get("result", [])
    if len(x) == 0 or len(y) == 0:
        return {} if per_label else 0
    if len(x) != len(y):
        return {} if per_label else 0
    else:
        if per_label:
            result = dict()
            # temp counters
            results = defaultdict(int)
            counts = defaultdict(int)
            try:
                for i in range(len(x)):
                    t = x[i]['type']
                    # extract label from annotation
                    labels = x[i]['value'].get(t)
                    if labels is None:
                        labels = x[i]['value'].get('hypertextlabels') or x[i]['value'].get('htmllabels')
                    for label in labels:
                        # for taxonomy and other non-str labels
                        label = str(label)
                        y_indexes = list(range(len(y)))
                        if label_order_matters:
                            y_indexes = [i]
                        for y_index in y_indexes:
                            if x[i]['value'] == y[y_index]['value']:
                                results[label] += 1
                                break
                        counts[label] += 1
                for label in counts:
                    result[label] = results[label] / counts[label]
            except:
                result = {}
        else:
            result = 0
            for i in range(len(x)):
                y_indexes = list(range(len(y)))
                if label_order_matters:
                    y_indexes = [i]
                for y_index in y_indexes:
                    if x[i]['value'] == y[y_index]['value']:
                        result += 1
                        break
            result = result / len(x)
    return result
