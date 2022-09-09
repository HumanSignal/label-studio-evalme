import pytest

from evalme.classification import ClassificationEvalItem, ChoicesEvalItem, naive, exact_matching_choices

from evalme.metrics import Metrics

from evalme.text.text import datetime_match, numbers_match

Metrics.register(
    name="datetime naive",
    form='empty_form',
    tag='datetime',
    func=datetime_match,
    desc='Basic matching function for datetime',
    is_default=True
)


Metrics.register(
    name="number simple",
    form='empty_form',
    tag='number',
    func=numbers_match,
    desc='Exact matching function for number',
    is_default=True
)


def test_config_with_several_control_types():
    """
    Test Metrics apply with different control types
    """
    result_1 = [
        {
          "id": "QfA7XYVj54",
          "type": "labels",
          "value": {
            "end": 76,
            "start": 68,
            "labels": [
              "Stool Frequency"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "QfA7XYVj54",
          "type": "number",
          "value": {
            "end": 76,
            "start": 68,
            "number": 1
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "number"
        },
        {
          "id": "QfA7XYVj54",
          "type": "choices",
          "value": {
            "end": 76,
            "start": 68,
            "choices": [
              "Adult content"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        },
        {
          "id": "QfA7XYVj54",
          "type": "datetime",
          "value": {
            "end": 76,
            "start": 68,
            "datetime": "2022-09-01"
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "datetime"
        },
        {
          "id": "QfA7XYVj54",
          "type": "textarea",
          "value": {
            "end": 76,
            "text": [
              "1"
            ],
            "start": 68
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "num_stools"
        },
        {
          "id": "QfA7XYVj54",
          "type": "textarea",
          "value": {
            "end": 76,
            "text": [
              "11/11/2022"
            ],
            "start": 68
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "date_bm"
        }
      ]
    result_2 = [
        {
          "id": "XBQRials-h",
          "type": "labels",
          "value": {
            "end": 76,
            "start": 68,
            "labels": [
              "yrdd"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "XBQRials-h",
          "type": "number",
          "value": {
            "end": 76,
            "start": 68,
            "number": 2
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "number"
        },
        {
          "id": "XBQRials-h",
          "type": "choices",
          "value": {
            "end": 76,
            "start": 68,
            "choices": [
              "Weapons"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        },
        {
          "id": "XBQRials-h",
          "type": "datetime",
          "value": {
            "end": 76,
            "start": 68,
            "datetime": "2022-09-02"
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "datetime"
        },
        {
          "id": "XBQRials-h",
          "type": "textarea",
          "value": {
            "end": 76,
            "text": [
              "2"
            ],
            "start": 68
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "num_stools"
        },
        {
          "id": "XBQRials-h",
          "type": "textarea",
          "value": {
            "end": 76,
            "text": [
              "12/12/2022"
            ],
            "start": 68
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "date_bm"
        }
      ]
    r1 = Metrics.apply({}, result_1, result_2)
    r2 = Metrics.apply({}, result_2, result_1)
    assert r1 == 0.0
    assert r2 == 0.0
