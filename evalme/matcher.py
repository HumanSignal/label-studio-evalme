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
                 token=None, project=1):
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
        self._load_data()

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

    def get_score(self):
        """
        One evaluation score per N predictions vs all annotations
        :return: agreement float[0..1] or None
        """
        score = 0
        tasks = 0
        for item in self._raw_data:
            annotations = item['annotations']
            predictions = item['predictions']
            score += matching_score(annotations, predictions)
            tasks += 1
        if tasks > 0:
            agreement = score / tasks
        else:
            agreement = None
        return agreement

    def get_score_per_prediction(self):
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
                scores[prediction['id']] = None
                score = 0
                tasks = 0
                for annotation in annotations:
                    try:
                        matching = Metrics.apply(
                            {}, prediction['result'], annotation['result'], symmetric=True, per_label=False
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
                    scores[prediction['id']] = score / tasks
        return scores


def matching_score(annotations, predictions):
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
                    {}, prediction['result'], annotation['result'], symmetric=True, per_label=False
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
