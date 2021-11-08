import itertools
from copy import deepcopy
from functools import partial
from collections import defaultdict

from evalme.eval_item import EvalItem
from evalme.utils import texts_similarity, get_text_comparator, parse_config_to_json
import logging
logger = logging.getLogger(__name__)

class TextTagsEvalItem(EvalItem):

    SHAPE_KEY = 'labels'

    def spans_iou(self, x, y):
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
                prefix = pred_value.get('when_label_value', '')
                if prefix:
                    prefix += ':'
                if self._shape_key in pred_value:
                    shape = self._shape_key
                else:
                    shape = self.SHAPE_KEY
                for l in pred_value[shape]:
                    total_score[prefix + l] += best_matching_score
                    total_weight[prefix + l] += 1
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

    def _spans_iou_by_start_end_offsets(self, x, y):
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

    def _spans_iou_by_text(self, x, y):
        text1 = x['text']
        text2 = y['text']
        # if one of the texts are empty return 0 match
        if len(text1) == 0 or len(text2) == 0:
            return 0

        # check if text is a full part of another annotation
        if text1 in text2:
            return len(text1) / len(text2)
        if text2 in text1:
            return len(text2) / len(text1)

        def _most_common_ends(text1, text2):
            # check if most common part of second text start is the end of text1
            ind = 0
            for i in range(1, len(text2)):
                if text2[:i] in text1:
                    ind = i
                    continue
                else:
                    break
            if text1.endswith(text2[:ind]):
                return ind / (len(text1) + len(text2) - ind)
            else:
                return -1

        iou1 = _most_common_ends(text1, text2)
        iou2 = _most_common_ends(text2, text1)
        m = max(iou1, iou2)
        if m > 0:
            return m
        else:
            return 0

    def spans_iou(self, x, y):
        # if labels are different returning 0 match
        x_set = set(x.get(self._shape_key, [])) or set(x.get('htmllabels', []))
        y_set = set(y.get(self._shape_key, [])) or set(y.get('htmllabels', []))
        if not (x_set == y_set):
            return 0

        # in case when substrings are presented - when can compute IOU using them
        if x.get('text') is not None and y.get('text') is not None:
            return self._spans_iou_by_text(x, y)
        # otherwise try using startOffset/endOffset for IOU
        else:
            return self._spans_iou_by_start_end_offsets(x, y)

    def _match(self, x, y, f):
        x_res = x.get(self._shape_key) or x.get('htmllabels')
        y_res = y.get(self._shape_key) or y.get('htmllabels')
        labels_match = texts_similarity(x_res, y_res, f)
        spans_match = self.spans_iou(x, y)
        return labels_match * spans_match

class TextAreaEvalItem(EvalItem):
    SHAPE_KEY = 'text'

    def match(self, item, algorithm='Levenshtein', qval=1):
        comparator = get_text_comparator(algorithm, qval)
        all_scores = []
        for gt, pred in zip(self.get_values_iter(), item.get_values_iter()):
            all_scores.append(texts_similarity(gt[self._shape_key], pred[self._shape_key], comparator))
        return sum(all_scores) / max(len(all_scores), 1)


class TaxonomyEvalItem(EvalItem):
    SHAPE_KEY = 'taxonomy'

    def spans_match(self, prediction, per_label=False):
        """
        Simple matching of labels in taxonomy
        """
        gt = self.get_values_iter()
        pred = prediction.get_values_iter()
        matches = 0
        not_found = 0
        if per_label:
            return {}
        else:
            for item_pred in pred:
                for item_gt in gt:
                    if item_gt == item_pred:
                        matches += 1
                        break
                else:
                    not_found += 1
            return matches / max((matches + not_found), 1)

    def spans_iou(self, prediction, per_label=False, label_config=None, label_weights=dict()):
        """
        Matching of taxonomy labels depending on content
        """
        if label_config is None:
            logger.warning("No label config - returning simple score.")
            return self.spans_match(prediction, per_label=per_label)

        master_tree = TaxonomyEvalItem._tree(label_config)

        gt = self.get_values_iter()
        pred = prediction.get_values_iter()
        matches = 0
        tasks = 0
        if per_label:
            results = {}
            for item_pred in pred:
                for item_gt in gt:
                    taxonomy_pred = item_pred['taxonomy']
                    taxonomy_gt = item_gt['taxonomy']
                    taxonomy_pred_list = list()
                    taxonomy_gt_list = list()
                    for item_pred_tx in taxonomy_pred:
                        taxonomy_pred_list.extend(TaxonomyEvalItem._transform_tree(master_tree, item_pred_tx))
                    for item_gt_tx in taxonomy_gt:
                        taxonomy_gt_list.extend(TaxonomyEvalItem._transform_tree(master_tree, item_gt_tx))
                    for item in taxonomy_pred_list:
                        if item in taxonomy_gt_list:
                            results[str(item[-1])] = label_weights.get(str(item[-1]), 1)
            return results
        else:
            for item_pred in pred:
                for item_gt in gt:
                    taxonomy_pred = item_pred['taxonomy']
                    taxonomy_pred_list = []
                    taxonomy_gt = item_gt['taxonomy']
                    taxonomy_gt_list = []
                    if taxonomy_pred == taxonomy_gt:
                        matches += 1
                        tasks += 1
                        break
                    else:
                        temp = 0
                        for item_pred_tx in taxonomy_pred:
                            taxonomy_pred_list.extend(TaxonomyEvalItem._transform_tree(master_tree, item_pred_tx))
                        for item_gt_tx in taxonomy_gt:
                            taxonomy_gt_list.extend(TaxonomyEvalItem._transform_tree(master_tree, item_gt_tx))
                        for item in taxonomy_pred_list:
                            if item in taxonomy_gt_list:
                                temp += 1
                        matches += (temp / max(len(taxonomy_gt_list), 1))
                        tasks += 1
            return matches / max(tasks, 1)

    def path_matches(self, prediction, per_label=False, label_weights=dict()):
        gt = self.get_values_iter()
        pred = prediction.get_values_iter()
        if per_label:
            results = dict()
            for item_gt in gt:
                item_gt_labels = item_gt['taxonomy']
                for item_pred in pred:
                    item_pred_labels = item_pred['taxonomy']
                    for item in itertools.product(item_gt_labels, item_pred_labels):
                        if item[0] == item[1]:
                            results[item[0][-1]] = label_weights.get(item[0][-1], 1)
            return results
        else:
            matches = 0
            tasks = 0
            for item_gt in gt:
                score = 0
                item_gt_labels = item_gt['taxonomy']
                for item_pred in pred:
                    item_pred_labels = item_pred['taxonomy']
                    for item in itertools.product(item_gt_labels, item_pred_labels):
                        score = max(score, TaxonomyEvalItem._compare_list(item[0], item[1]))
                tasks += 1
                matches += score
            return matches / max(tasks, 1)

    @staticmethod
    def _tree(label_config):
        """
        Creating Tree from label_config
        """

        def recursive_lookup(d, k='Taxonomy'):
            if not isinstance(d, dict):
                return None
            if k in list(d.keys()):
                return d[k]
            for v in d.values():
                if isinstance(v, dict):
                    a = recursive_lookup(v, k)
                    if a is not None:
                        return a
                if isinstance(v, list):
                    for item in v:
                        a = recursive_lookup(item, k)
                        if a is not None:
                            return a
            return None

        temp = parse_config_to_json(label_config)
        res = recursive_lookup(temp)
        tree = TaxonomyEvalItem._subtree(res.get('Choice'))
        return tree

    @staticmethod
    def _subtree(node):
        """
        Create dict() from Tree
        """
        if node is None:
            return dict()
        subtree = dict()
        if isinstance(node, list):
            for subnode in node:
                subtree[subnode['@value']] = TaxonomyEvalItem._subtree(subnode.get('Choice'))
        else:
            subtree[node['@value']] = TaxonomyEvalItem._subtree(node.get('Choice'))
        return subtree

    @staticmethod
    def _transform_tree(config, node):
        """
        Transform tree to paths list
        """
        import operator
        from functools import reduce

        def paths(self):
            """
            Get all paths from Tree
            """
            if not self.keys():
                return []
            local_paths = []
            for child in self:
                child_paths = paths(self[child])
                if child_paths:
                    for path in child_paths:
                        local_paths.append([child] + path)
                else:
                    local_paths.append([child])
            return local_paths

        nodes = []

        items = reduce(operator.getitem, node, config)
        path_list = paths(items)
        if path_list:
            for path in path_list:
                temp_node = deepcopy(node)
                temp_node.extend(path)
                nodes.append(temp_node)
        else:
            nodes.append(node)

        return nodes

    @staticmethod
    def _compare_list(gt, pred):
        score = 0
        for p, g in zip(pred, gt):
            if p == g:
                score += 1
            else:
                break
        return score / max(len(gt), 1)


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


def _as_taxonomy_eval_item(item):
    if not isinstance(item, TaxonomyEvalItem):
        return TaxonomyEvalItem(item)
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


def intersection_taxonomy(item_gt, item_pred, label_weights=dict(), per_label=False, label_config=None):
    item_gt = _as_taxonomy_eval_item(item_gt)
    item_pred = _as_taxonomy_eval_item(item_pred)
    return item_gt.spans_iou(item_pred, per_label=per_label, label_config=label_config, label_weights=label_weights)


def path_match_taxonomy(item_gt, item_pred, label_weights=dict(), per_label=False):
    item_gt = _as_taxonomy_eval_item(item_gt)
    item_pred = _as_taxonomy_eval_item(item_pred)
    return item_gt.path_matches(item_pred, per_label=per_label, label_weights=label_weights)