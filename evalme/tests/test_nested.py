from functools import partial

import pytest

from evalme.metrics import Metrics

from evalme.image.object_detection import iou_bboxes_textarea, iou_polygons_textarea
from evalme.text.text import intersection_textarea_tagging, match_textareas, intersection_text_tagging

Metrics.register(
    name='1d_region_intersection_threshold',
    form='iou_threshold',
    tag='Labels',
    func=intersection_text_tagging,
    desc='Percentage of matched regions by IOU w.r.t threshold'
)

Metrics.register(
    name='edit_distance',
    form='edit_distance',
    tag='TextArea',
    func=match_textareas,
    desc='Text edit distance'
)

Metrics.register(
    name='edit_distance_per_span',
    form='edit_distance',
    tag='TextArea[per_region=span]',
    func=partial(intersection_textarea_tagging, shape_key='text'),
    desc='Text edit distance per span region'
)

Metrics.register(
    name='edit_distance_per_span',
    form='iou_threshold',
    tag='TextArea[per_region=span]',
    func=partial(intersection_textarea_tagging, shape_key='text'),
    desc='Text edit distance per span region, with percentage of matched spans by IOU w.r.t threshold'
)

Metrics.register(
    name='edit_distance_per_hyperspan',
    form='iou_threshold',
    tag='TextArea[per_region=hyperspan]',
    func=partial(intersection_textarea_tagging, shape_key='text'),
    desc='Text edit distance per hypertext span region, with percentage of matched spans by IOU w.r.t threshold'
)

Metrics.register(
    name='edit_distance_per_bbox',
    form='edit_distance',
    tag='TextArea[per_region=bbox]',
    func=partial(iou_bboxes_textarea, shape_key='text'),
    desc='Text edit distance per bbox region'
)

Metrics.register(
    name='edit_distance_per_polygon',
    form='edit_distance',
    tag='TextArea[per_region=poly]',
    func=partial(iou_polygons_textarea, shape_key='text'),
    desc='Text edit distance per polygon region'
)

@pytest.mark.parametrize('labels_weight,expected_score', [(0, 1), (1, 0.6666666666666666)])
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
                "text": ["sample text"]
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
                    "text": ["sample text"]
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
                    "text": ["sample text"]
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
                        "sample text"
                    ]
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
