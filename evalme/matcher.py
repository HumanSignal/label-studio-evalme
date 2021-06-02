from collections import defaultdict

import numpy as np
import requests
import json
import logging

from evalme.image.object_detection import prediction_bboxes, matrix_iou_bboxes
from evalme.metrics import Metrics
from evalme.utils import calculate_ap

logger = logging.getLogger(__name__)


class Matcher:
    """
    Class for loading data from label studio
    """
    def __init__(self, url='http://127.0.0.1:8000',
                 token='', project=1):
        """
        :param url: Label studio url
        :param token: access token
        :param project: from which project to load data
        """
        self._headers = {
            'Authorization': 'Token ' + token,
            'Content-Type': 'application/json'
        }
        self._raw_data = {}
        self._export_url = url + f'/api/projects/{project}/export?exportType=JSON'
        self._project_url = url + f'/api/projects/{project}'
        self._control_weights = {}

    def _load_data(self):
        response = requests.get(self._export_url, headers=self._headers)
        control_weights = requests.get(self._project_url, headers=self._headers)
        self._control_weights = json.loads(control_weights.text)
        self._raw_data = json.loads(response.text)

    def refresh(self):
        self._load_data()

    def _load_from_file(self, filename):
        with open(filename) as f:
            self._raw_data = json.load(f)

    def load(self, filename):
        self._load_from_file(filename)

    def get_iou_score(self):
        """
        One evaluation score per N predictions vs all annotations
        :return: agreement float[0..1] or None
        """
        score = 0
        tasks = 0
        for item in self._raw_data:
            annotations = item['annotations']
            predictions = item['predictions']
            score += self.matching_score(annotations, predictions, metric_name='iou_bboxes')
            tasks += 1
        if tasks > 0:
            agreement = score / tasks
        else:
            agreement = None
        return agreement

    def get_score_per_task(self, metric_name=None):
        """
        One evaluation score per N predictions vs all annotations
        :return: agreement float[0..1] or None
        """
        agreement = []

        for item in self._raw_data:
            annotations = item['annotations']
            predictions = item['predictions']
            score = self.matching_score(annotations, predictions, metric_name=metric_name)
            agreement.append(score)
        return agreement

    def get_score_per_prediction(self, per_label=False, metric_name=None):
        """
        N agreement scores per each prediction vs corresponding annotation
        :return: dict  {
                        prediction.id: float[0..1]
                        }
        """
        scores = {}
        control_weights = self._control_weights or {}
        for item in self._raw_data:
            annotations = item['annotations']
            predictions = item['predictions']
            for prediction in predictions:
                if per_label:
                    scores[prediction['id']] = defaultdict(int)
                    score = defaultdict(int)
                    tasks = defaultdict(int)
                else:
                    scores[prediction['id']] = None
                    score = 0
                    tasks = 0
                for annotation in annotations:
                    try:
                        matching = Metrics.apply(
                            control_weights, prediction['result'], annotation['result'],
                            symmetric=True, per_label=per_label,
                            metric_name=metric_name
                        )
                        if per_label:
                            for label in matching:
                                score[label] += matching[label]
                                tasks[label] += 1
                        else:
                            score += matching
                            tasks += 1
                    except Exception as exc:
                        logger.error(
                            f"Can\'t compute matching score in similarity matrix for task=,"
                            f"annotation={annotation}, prediction={prediction}, "
                            f"Reason: {exc}",
                            exc_info=True,
                        )
                if per_label:
                    for label in tasks:
                        scores[prediction['id']][label] = score[label] / tasks[label]
                else:
                    if tasks > 0:
                        scores[prediction['id']] = score / tasks
        return scores

    def agreement_matrix(self, per_label=False, metric_name=None):
        """
        Per task agreement matrix for annotations
        :return: { task.id: np.array(m, m) -> float[0..1]
        }
        """
        agreement = {}
        control_weights = self._control_weights or {}
        for item in self._raw_data:
            annotations = item['annotations']
            num_results = len(annotations)
            matrix = np.full((num_results, num_results), np.nan)
            for i in range(num_results):
                for j in range(i + 1, num_results):
                    matching_score = Metrics.apply(
                        control_weights,
                        annotations[i],
                        annotations[j],
                        symmetric=True,
                        per_label=per_label,
                        metric_name=metric_name,
                    )
                    matrix[i][j] = matrix[j][i] = matching_score
            agreement[item['id']] = matrix
        return agreement

    def matching_score(self, annotations, predictions, metric_name=None, iou_threshold=None):
        """
        One evaluation score per N predictions vs all annotations per task
        :return: agreement float[0..1] or None
        """
        score = 0
        tasks = 0
        control_weights = self._control_weights or {}
        for annotation in annotations:
            for prediction in predictions:
                try:
                    matching = Metrics.apply(
                        control_weights, prediction['result'], annotation['result'], symmetric=True, per_label=False,
                        metric_name=metric_name, iou_threshold=iou_threshold
                    )
                    score += matching
                    tasks += 1
                except Exception as exc:
                    logger.error(
                        f"Can\'t compute matching score in similarity matrix for task=,"
                        f"annotation={annotation}, prediction={prediction}, "
                        f"Reason: {exc}",
                        exc_info=True,
                    )
        if tasks > 0:
            return score / tasks
        else:
            return None

    def matching_score_per_label(self, annotations, predictions, metric_name=None, iou_threshold=None):
        """
        One evaluation score per N predictions vs all annotations per label
        :return: dict { label: float[0..1] or None }
        """
        score = defaultdict(int)
        tasks = defaultdict(int)
        control_weights = self._control_weights or {}
        for annotation in annotations:
            for prediction in predictions:
                try:
                    matching = Metrics.apply(
                        control_weights, prediction['result'], annotation['result'], symmetric=True, per_label=True,
                        metric_name=metric_name, iou_threshold=iou_threshold
                    )
                    for label in matching:
                        score[label] += matching[label]
                        tasks[label] += 1
                except Exception as exc:
                    logger.error(
                        f"Can\'t compute matching score in similarity matrix for task=,"
                        f"annotation={annotation}, prediction={prediction}, "
                        f"Reason: {exc}",
                        exc_info=True,
                    )
        results = {}
        for label in score:
            results[label] = score[label] / tasks[label]
        return results

    def get_mAP_score(self):
        """
        One mAP score per N predictions vs all annotations
        :return: agreement float[0..1] or None
        """
        items_ars = []
        for item in self._raw_data:
            annotations = item['annotations']
            predictions = item['predictions']
            for prediction in predictions:
                pred_results = []
                for annotation in annotations:
                    try:
                        matching = prediction_bboxes(annotation['result'], prediction['result'], iou_threshold=0.5,)
                        pred_results.append(matching)
                    except Exception as e:
                        print(e)
                items_ars.append(calculate_ap(pred_results))
        return sum(items_ars) / len(items_ars)

    def get_results_comparision_matrix_iou(self):
        """
        Total IOU matrix for each shape in annotations
        """
        results = {}
        for task in self._raw_data:
            annotations = task['annotations']
            predictions = task['predictions']
            results[task['id']] = self.get_results_comparision_matrix_by_task_iou(annotations, predictions)
        return results

    def get_results_comparision_matrix_by_task_iou(self, annotations, predictions):
        results = {}
        label_weights = self._control_weights.get('control_weights', {}).get('label', {}).get('labels')
        for annotation in annotations:
            results[annotation['id']] = {}
            for prediction in predictions:
                results[annotation['id']][prediction['id']] = {}
                try:
                    t = matrix_iou_bboxes(annotation['result'], prediction['result'],
                                          label_weights=label_weights)
                    results[annotation['id']][prediction['id']] = t
                except Exception as exc:
                    logger.error(
                        f"Can\'t compute matching score in similarity matrix for task=,"
                        f"annotation={annotation}, prediction={prediction}, "
                        f"Reason: {exc}",
                        exc_info=True,
                    )
        return results

    def get_annotations_agreement(self, metric_name=None):
        """
        One evaluation score per all annotations
        :return: agreement float[0..1] or None
        """
        score = 0
        tasks = 0
        for item in self._raw_data:
            annotations = item['annotations']
            score += self.matching_score(annotations, annotations, metric_name=metric_name)
            tasks += 1
        if tasks > 0:
            agreement = score / tasks
        else:
            agreement = None
        return agreement
