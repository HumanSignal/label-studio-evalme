import numpy as np
import attr

from scipy.cluster.hierarchy import single, complete, fcluster
from scipy.spatial.distance import squareform
from itertools import chain
from collections import defaultdict
from copy import deepcopy

from evalme.classification import exact_matching_choices
from evalme.image.object_detection import iou_bboxes
from evalme.text import intersection_text_tagging
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
    def apply(cls, project, result_first, result_second, symmetric=False, per_label=False):
        """
        Compute matching score between first and second completion results
        Args:
            project: Project object used for getting matching score function parameters
            result_first: first completion.result
            result_second: second completion.result
            symmetric: symmetric result doesn't depend on the first/second results order

        Returns:
            Matching score averaged over all different "from_name"s with corresponding weights taken from project.control_weights  # noqa
        """

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
            all_controls[r['from_name']] = cls.get_type(r)

        def get_matching_func(control_type):
            # TODO support metric_name
            # if project.metric_name and len(all_controls) == 1:
            #     # user specified which matching score function to use, supported only with one control tag
            #     return cls._metrics.get(project.metric_name)
            return cls.get_default_metric_for_tag(control_type)

        def symmetrize(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        # TODO fix metric_params
        # params = project.metric_params or {}
        params = {}
        if per_label:
            score, n = defaultdict(int), defaultdict(int)
        else:
            score, n = 0, 0

        # aggregate matching scores over all existed controls
        for control_name, control_type in all_controls.items():
            control_weights = project.get("control_weights", {})
            control_weights = control_weights.get(control_name, {})
            overall_weight = control_weights.get('overall', 1)
            label_weights = control_weights.get('labels')
            control_params = deepcopy(params)
            control_params['label_weights'] = label_weights
            control_params['per_label'] = per_label

            matching_func = get_matching_func(control_type)
            if not matching_func:
                raise NotImplementedError(f'No matching function found for control type {control_type} in {project}')

            results_first_by_from_name = cls.filter_results_by_from_name(result_first, control_name)
            results_second_by_from_name = cls.filter_results_by_from_name(result_second, control_name)
            s = matching_func.func(results_first_by_from_name, results_second_by_from_name, **control_params)
            if symmetric:
                s_reversed = matching_func.func(results_second_by_from_name, results_first_by_from_name, **control_params)
                if per_label:
                    for l in set(list(s.keys()) + list(s_reversed.keys())):
                        s[l] = symmetrize(s.get(l), s_reversed.get(l))
                else:
                    s = symmetrize(s, s_reversed)

            if per_label:
                for l in s:
                    score[l] += s[l] * overall_weight
                    n[l] += overall_weight
            else:
                score += s * overall_weight
                n += overall_weight

        def clipped(s):
            if s > 1 or s < 0:
                logger.error(f'Error in project {project}. Matching score {s} is not within [0, 1] interval '
                             f'for the following results:\nx={result_first}\ny={result_second}. It will be clipped')
                return 0 if s < 0 else 1
            return s

        if per_label:
            for l in score:
                if n[l] > 0:
                    score[l] /= float(n[l])
                else:
                    score[l] = 0
                score[l] = clipped(score[l])
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
        :param results: list of results
        :param threshold: the comparison between 2 results should be above this threshold to join them in one group
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
    name='iou_bboxes',
    form=None,
    tag='RectangleLabels',
    func=iou_bboxes,
    desc='IOU for bounding boxes'
)

Metrics.register(
    name='1d_region_intersection',
    form=None,
    tag='Labels',
    func=intersection_text_tagging,
    desc='Intersection over 1D text spans'
)

Metrics.register(
    name='exact_match_choices',
    form=None,
    tag='Choices',
    func=exact_matching_choices,
    desc='Exact matching choices'
)