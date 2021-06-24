import pytest

from evalme.image.object_detection import KeyPointsEvalItem, keypoints_distance


def test_keypoints_matching():
    '''
    Matching with almost 1 distance
    '''
    test_data = [
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 35.111111,
                "y": 65.41666666666667,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }],
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 34.222222,
                "y": 64.4167,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }]
    ]
    assert keypoints_distance(test_data[0], test_data[1], label_weights={}) == 1


def test_keypoints_matching_per_label():
    '''
    Matching with almost 1 distance per label
    '''
    test_data = [
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 35.111111,
                "y": 65.41666666666667,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }],
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 34.222222,
                "y": 64.4167,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }]
    ]
    assert keypoints_distance(test_data[0], test_data[1], label_weights={}, per_label=True) == {"Engine": 1}


def test_keypoints_not_matching():
    '''
    Not Matching with almost 1 distance
    '''
    test_data = [
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 35.333333,
                "y": 65.41666666666667,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }],
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 34.222222,
                "y": 64.4165,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }]
    ]

    assert keypoints_distance(test_data[0], test_data[1], label_weights={}) == 0

def test_keypoints_not_matching_label():
    '''
    Not Matching label
    '''
    test_data = [
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 34.0,
                "y": 64.0,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }],
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 34.0,
                "y": 64.0,
                "width": 0.625,
                "keypointlabels": ["Engine1"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }]
    ]
    assert keypoints_distance(test_data[0], test_data[1], label_weights={}) == 0


def test_keypoints_not_matching_per_label():
    '''
    Not Matching with almost 1 distance per label
    '''
    test_data = [
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 35.333333,
                "y": 65.41666666666667,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }],
        [{
            "id": "S6oszbKrqK",
            "type": "keypointlabels",
            "value": {
                "x": 34.222222,
                "y": 64.4165,
                "width": 0.625,
                "keypointlabels": ["Engine"]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 320,
            "original_height": 240
        }]
    ]
    assert keypoints_distance(test_data[0], test_data[1], label_weights={}, per_label=True) =={"Engine": 0}