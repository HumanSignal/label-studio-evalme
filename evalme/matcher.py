from collections import defaultdict

import requests
import json
import logging

from evalme.metrics import Metrics

logger = logging.getLogger(__name__)


class Matcher:
    """
    Class for loading data from label studio
    """
    def __init__(self, url='http://127.0.0.1:8000',
                 token='', project=1):
        """
        :param url:
        :param token:
        :param project:
        """
        self._headers = {
            'Authorization': 'Token ' + token,
            'Content-Type': 'application/json'
        }
        self._raw_data = {}
        self._export_url = url + f'/api/projects/{project}/export?exportType=JSON'

    def _load_data(self):
        response = requests.get(self._export_url, headers=self._headers)
        self._raw_data = json.loads(response.text)

    def refresh(self):
        self._load_data()

    def _load_from_file(self, filename):
        with open(filename) as f:
            self._raw_data = json.load(f)

    def load(self, filename):
        self._load_from_file(filename)

    def get_iou_score(self, per_label=False, threshold=0.8):
        """
        One evaluation score per N predictions vs all annotations
        :return: agreement float[0..1] or None
        """
        score = 0
        tasks = 0
        for item in self._raw_data:
            annotations = item['annotations']
            predictions = item['predictions']
            score += matching_score(annotations, predictions, metric_name='iou_bboxes')
            tasks += 1
        if tasks > 0:
            agreement = score / tasks
        else:
            agreement = None
        return agreement

    def get_score_per_task(self, per_label=False):
        """
        One evaluation score per N predictions vs all annotations
        :return: agreement float[0..1] or None
        """
        agreement = []

        for item in self._raw_data:
            score = 0
            tasks = 0
            annotations = item['annotations']
            predictions = item['predictions']
            score += matching_score(annotations, predictions)
            tasks += 1
            if tasks > 0:
                agreement.append(score / tasks)
        return agreement

    def get_score_per_prediction(self, per_label=False):
        """
        N agreement scores per each prediction vs corresponding annotation
        :return: dict  {
                        prediction.id: float[0..1]
                        }
        """
        scores = {}
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
                            {}, prediction['result'], annotation['result'], symmetric=True, per_label=per_label
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

    def get_mAP_score(self, per_label=False, threshold=0.8):
        """
        One mAP score per N predictions vs all annotations
        :return: agreement float[0..1] or None
        """
        score = 0
        tasks = 0
        for item in self._raw_data:
            annotations = item['annotations']
            predictions = item['predictions']
            score += matching_score(annotations, predictions, metric_name='mAP_bboxes', iou_threshold=threshold)
            tasks += 1
        if tasks > 0:
            agreement = score / tasks
        else:
            agreement = None
        return agreement


def matching_score(annotations, predictions, metric_name=None, iou_threshold=None):
    """
    One evaluation score per N predictions vs all annotations per task
    :return: agreement float[0..1] or None
    """
    score = 0
    tasks = 0
    for annotation in annotations:
        for prediction in predictions:
            try:
                matching = Metrics.apply(
                    {}, prediction['result'], annotation['result'], symmetric=True, per_label=False, metric_name=metric_name, iou_threshold=iou_threshold
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

if __name__ == '__main__':
    ls = Matcher(token='984dbd6702a7df0429703a76afb3b7bc66477d27', project=7)
    #ls.refresh()
    ls.load(r"C:\Temp\Screens\temp\test_tasks.json")
    print(ls._raw_data)
    t = ls.get_mAP_score(threshold=0.5)
    print(t)