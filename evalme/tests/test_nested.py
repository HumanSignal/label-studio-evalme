import pytest

from evalme.metrics import Metrics


@pytest.mark.parametrize('labels_weight,expected_score', [(0, 1), (1, 0.83)])
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
                "value": {
                    "labels": ["Label"]
                }
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
                "value": {
                    "text": "sample text"
                }
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
                    "value": {
                        "labels": ["Label"]
                    }
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
                    "value": {
                        "text": "sample text"
                    }
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
                "value": {
                    "labels": ["Label"]
                }
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
                    "value": {
                        "text": "sample text"
                    }
                }
            }, {
            "id": "67890",
            "from_name": "labels",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": 25,
                "end": 35,
                "value": {
                    "labels": ["Label"]
                }
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
                    "value": {
                        "text": "sample text"
                    }
                }
            }]
    ]
    score = Metrics.apply(
        {'control_weights': {'labels': {'overall': labels_weight, 'type': 'Labels'}}},
        test_data[0], test_data[1],
        metric_name='default',
        feature_flags={'ff_back_dev_2762_textarea_weights_30062022_short': ''}
    )
    assert score == expected_score
