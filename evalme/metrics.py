import inspect

import numpy as np
import attr

from scipy.cluster.hierarchy import single, complete, fcluster
from scipy.spatial.distance import squareform
from itertools import chain
from collections import defaultdict
from copy import deepcopy

from evalme.classification import naive
import logging

logger = logging.getLogger(__name__)


@attr.s
class MetricWrapper(object):
    name = attr.ib()
    form = attr.ib()
    description = attr.ib()
    func = attr.ib()
    tag = attr.ib()
    is_default = attr.ib(default=True)


class Metrics(object):

    _metrics = {}

    @classmethod
    def _norm_tag(cls, tag):
        return tag.lower()

    @classmethod
    def register(cls, name, form, tag, func, desc, is_default=True):
        cls._metrics[name] = MetricWrapper(name, form, desc, func, cls._norm_tag(tag), is_default)

    @classmethod
    def get_schema(cls, tag):
        return [m for m in cls._metrics.values() if m.tag == cls._norm_tag(tag)]

    @classmethod
    def get_form_by_name(cls, name):
        m = cls._metrics.get(name)
        if m:
            return m.form

    @classmethod
    def get_default_metric_for_tag(cls, tag):
        return next((m for m in cls._metrics.values() if m.tag == tag and m.is_default), None)

    @classmethod
    def get_default_metric_for_name_tag(cls, tag, name):
        metric = cls._metrics.get(name)
        if (metric is not None) and ((metric.tag == tag) or (metric.tag == 'all')):
            return metric
        else:
            return cls.get_default_metric_for_tag(tag)

    @classmethod
    def filter_results_by_from_name(cls, results, from_name):
        return list(filter(lambda r: r.get('from_name') == from_name, results))

    @classmethod
    def get_type(cls, result):
        t = result.get('type')
        # check for per_region conditions
        if t in ('choices', 'textarea'):
            if 'start' in result['value'] and 'end' in result['value']:
                t += '[per_region=span]'
            elif 'x' in result['value'] and 'y' in result['value']:
                t += '[per_region=bbox]'
            elif 'points' in result['value']:
                t += '[per_region=poly]'
        return t

    @classmethod
    def apply(cls, project, result_first, result_second, symmetric=True, per_label=False,
              metric_name=None, iou_threshold=None):
        """
        Compute matching score between first and second completion results
        Args:
        :param project: Project object used for getting matching score function parameters
        :param result_first: first completion.result
        :param result_second: second completion.result
        :param symmetric: symmetric result doesn't depend on the first/second results order
        :param metric_name: name of metric to use
        :param per_label: per_label calculation or overall
        :param iou_threshold: intersection over union threshold
        Returns:
            Matching score averaged over all different "from_name"s with corresponding weights taken from project.control_weights  # noqa
        """
        annotations_or_result = None
        # decide which object to use annotation-based or result-based
        if isinstance(result_first, dict) and isinstance(result_second, dict):
            annotations_or_result = True
        elif isinstance(result_first, list) and isinstance(result_second, list):
            annotations_or_result = False
        else:
            raise ValueError('Both inputs should be Annotation(dict) or Results(list)')

        if not result_first and not result_second:
            if per_label:
                return {}
            return float(type(result_first) == type(result_second))

        # collect mapping between control tag name and control type
        all_controls = {}
        for r in chain(result_first, result_second):
            if 'from_name' not in r:
                # we skip all non-control tag results like relations, etc.
                continue
            result_type = cls.get_type(r)
            if result_type == 'rectangle':
                # keep only rectangle if any and skip other control types
                all_controls = {result_type: result_type}
                break
            all_controls[r['from_name']] = result_type

        def get_matching_func(control_type, name=None):
            if name:
                return cls.get_default_metric_for_name_tag(control_type, name)
            else:
                return cls.get_default_metric_for_tag(control_type)

        def get_matching_func_annotation(name=None):
            return cls.get_default_metric_for_name_tag('all', name)

        def symmetrize(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)
        # get metric params
        params = project.get("metric_params", {})
        # remove backup parameters
        list(map(params.pop, [item for item in params if item.startswith("__")]))
        if per_label:
            score, n = defaultdict(int), defaultdict(int)
        else:
            score, n = 0, 0

        if annotations_or_result:
            # get matching score over annotations as a hole
            if project == {}:
                control_params = {}
            else:
                control_weights = project.get("control_weights", {})
                control_params = deepcopy(params)
                control_params['control_weights'] = control_weights
                control_params['per_label'] = per_label

            matching_func = get_matching_func_annotation(name=metric_name)
            if per_label:
                score, n = matching_func.func(result_first, result_second, **control_params)
            else:
                score = matching_func.func(result_first, result_second, **control_params)
                n = 1
        else:
            # aggregate matching scores over all existed controls
            for control_name, control_type in all_controls.items():
                control_weights = project.get("control_weights", {})
                control_weights = control_weights.get(control_name, {})
                overall_weight = control_weights.get('overall', 1)
                label_weights = control_weights.get('labels')
                control_params = deepcopy(params)
                control_params['label_weights'] = label_weights
                control_params['per_label'] = per_label
                if iou_threshold:
                    control_params['iou_threshold'] = iou_threshold

            matching_func = get_matching_func(control_type, metric_name)
            if not matching_func:
                logger.error(f'No matching function found for control type {control_type} in {project}.'
                             f'Using naive calculation.')
                matching_func = cls._metrics.get('naive')
            # identify if label config need
            func_args = inspect.getfullargspec(matching_func.func)
            if 'label_config' in func_args[0]:
                control_params['label_config'] = project.get("label_config")

            results_first_by_from_name = cls.filter_results_by_from_name(result_first, control_name)
            results_second_by_from_name = cls.filter_results_by_from_name(result_second, control_name)
            s = matching_func.func(results_first_by_from_name, results_second_by_from_name, **control_params)
            if symmetric:
                s_reversed = matching_func.func(results_second_by_from_name, results_first_by_from_name,
                                                **control_params)
                if per_label:
                    for label in set(list(s.keys()) + list(s_reversed.keys())):
                        s[label] = symmetrize(s.get(label), s_reversed.get(label))
                else:
                    s = symmetrize(s, s_reversed)

            if per_label:
                for label in s:
                    score[label] += s[label] * overall_weight
                    n[label] += overall_weight
            else:
                score += s * overall_weight
                n += overall_weight

        def clipped(s):
            if s > 1 or s < 0:
                logger.warning('Error in project %s. Matching score %s is not within [0, 1] interval '
                             'for the following results:\nx=%s\ny=%s. It will be clipped',
                             str(project), str(s), str(result_first), str(result_second))
                return 0 if s < 0 else 1
            return s

        if per_label:
            for label in score:
                if n[label] > 0:
                    score[label] /= float(n[label])
                else:
                    score[label] = 0
                score[label] = clipped(score[label])
            return score

        return clipped(score / float(n) if n > 0 else 0)

    @classmethod
    def average(cls, project, results):
        n = len(results)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0
        mean_score = 0
        for i in range(n):
            for j in range(i + 1, n):
                mean_score += Metrics.apply(project, results[i], results[j], symmetric=True)
        norm = n * (n - 1) / 2
        return mean_score / max(norm, 1)

    @classmethod
    def group(cls, project, results, threshold=0.0, use_single_linkage_by_default=False):
        """
        Returns dict with groups, where key are group index and values are list of results indices
        :param project: Project object used for getting matching score function parameters
        :param results: list of results
        :param threshold: the comparison between 2 results should be above this threshold to join them in one group
        :param use_single_linkage_by_default:
        :return: {0: [1,2,3], 1: [4,5], 2: [6]}
        """
        num_results = len(results)
        if num_results == 0:
            raise ValueError(f'Can\'t group empty results for project {project}')
        if num_results == 1:
            return {0: [0]}
        result_sim = np.zeros((num_results, num_results), dtype=np.float64)
        for i in range(num_results):
            for j in range(i + 1, num_results):
                result_sim[i, j] = Metrics.apply(project, results[i], results[j], symmetric=True)
        result_sim = result_sim + result_sim.T
        margin = 1.01 * np.max(result_sim)
        dists = margin - squareform(result_sim)
        if project.agreement_method == project.SINGLE or not project.agreement_method:
            linkage_matrix = single(dists)
        elif project.agreement_method == project.COMPLETE:
            linkage_matrix = complete(dists)
        else:
            if use_single_linkage_by_default:
                linkage_matrix = single(dists)
            else:
                raise ValueError(f'Unknown agreement method {project.agreement_method}')
        clusters = fcluster(linkage_matrix, t=margin - threshold, criterion='distance')
        groups = defaultdict(list)
        for i, cluster_idx in enumerate(clusters):
            groups[cluster_idx].append(i)
        return groups


Metrics.register(
    name='naive',
    form=None,
    tag='all',
    func=naive,
    desc='Naive comparison of result dict'
)


def get_agreement(annotation_from,
                  annotation_to,
                  project_params={},
                  per_label=False,
                  metric_name=None):
    """
    Calculate scores between 2 Annotation instances
    :param annotation_from: Ground truth annotation instance
    :param annotation_to: Predicted annotation instance
    :param project_params: Project parameters for matching score function [e.g. iou threshold]
    :param per_label: per_label calculation or overall
    :param metric_name: Registred metric name for matching function
    :return: For overall: float[0..1], for per_label tuple(Float[0..1], dict(Float[0..1]))
    """
    score = Metrics.apply(project_params,
                          annotation_from,
                          annotation_to,
                          metric_name=metric_name)
    if per_label:
        score_per_label = Metrics.apply(project_params,
                              annotation_from,
                              annotation_to,
                              per_label=True,
                              metric_name=metric_name)
        return score, score_per_label
    return score
