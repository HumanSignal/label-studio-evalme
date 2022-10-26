import pytest

from evalme.metrics import Metrics


@pytest.mark.parametrize('labels_weight,expected_score', [(0, 0.7), (1, 0.85)])
def test_basic_matching_function_nested_with_project_weights(labels_weight, expected_score):
    test_data = [
        [{
            "id": "12345",
            "from_name": "labels",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": 0,
                "end": 10,
                "labels": ["Label"]
            }
        },
        {
            "id": "12345",
            "from_name": "textarea",
            "to_name": "text",
            "type": "textarea",
            "value": {
                "start": 0,
                "end": 10,
                "text": ["common string AAAAAA"]
            }
        },
            {
                "id": "67890",
                "from_name": "labels",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": 20,
                    "end": 30,
                    "labels": ["Label"]
                }
            },
            {
                "id": "67890",
                "from_name": "textarea",
                "to_name": "text",
                "type": "textarea",
                "value": {
                    "start": 20,
                    "end": 30,
                    "text": ["common string BBBBBB"]
                }
            }
        ],
        [{
            "id": "12345",
            "from_name": "labels",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": 5,
                "end": 15,
                "labels": ["Label"]
            }
        },
            {
                "id": "12345",
                "from_name": "textarea",
                "to_name": "text",
                "type": "textarea",
                "value": {
                    "start": 5,
                    "end": 15,
                    "text": ["common string BBBBBB"]
                }
            }, {
            "id": "67890",
            "from_name": "labels",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": 25,
                "end": 35,
                "labels": ["Label"]
            }
        },
            {
                "id": "67890",
                "from_name": "textarea",
                "to_name": "text",
                "type": "textarea",
                "value": {
                    "start": 25,
                    "end": 35,
                    "text": [
                        "common string AAAAAA"
                    ]
                }
            }]
    ]
    score = Metrics.apply(
        {'control_weights': {'labels': {'overall': labels_weight, 'type': 'Labels'}}},
        test_data[0], test_data[1],
        metric_name='default',
        feature_flags={'ff_back_dev_2762_textarea_weights_30062022_short': True}
    )
    assert score == expected_score
