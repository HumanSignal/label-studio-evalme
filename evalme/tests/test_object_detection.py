import pytest

from evalme.image.object_detection import KeyPointsEvalItem, keypoints_distance, PolygonObjectDetectionEvalItem, OCREvalItem


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


def test_object_detection_fixing_polygon():
    points = [[37.5, 23.046875], [36.328125, 23.828125], [35.15625, 25.0], [37.109375, 23.4375], [38.671875, 23.046875], [41.015625, 25.390625], [41.015625, 26.953125], [40.625, 34.375], [40.234375, 37.890625], [41.40625, 39.453125], [41.796875, 40.625], [42.1875, 41.796875], [42.96875, 42.96875], [39.453125, 43.359375], [39.453125, 41.796875], [38.28125, 41.015625], [39.0625, 42.1875], [38.28125, 44.140625], [36.71875, 43.359375], [37.109375, 41.796875], [36.71875, 40.625], [37.5, 39.453125], [36.328125, 38.28125], [35.15625, 39.0625], [34.765625, 40.234375], [34.375, 41.40625], [33.984375, 42.96875], [33.203125, 44.140625], [32.8125, 45.3125], [32.421875, 46.484375], [31.640625, 47.65625], [31.25, 48.828125], [31.25, 50.390625], [30.859375, 51.5625], [30.078125, 52.734375], [29.6875, 53.90625], [29.296875, 55.078125], [28.90625, 56.25], [29.296875, 57.8125], [38.28125, 57.8125], [42.96875, 58.203125], [56.25, 57.8125], [59.765625, 58.203125], [60.9375, 57.421875], [60.546875, 55.859375], [60.15625, 54.296875], [59.765625, 53.125], [59.765625, 48.828125], [59.375, 46.875], [58.984375, 44.921875], [58.203125, 43.75], [57.8125, 42.578125], [57.421875, 41.40625], [57.421875, 37.890625], [57.03125, 35.9375], [56.640625, 34.765625], [56.25, 31.640625], [55.859375, 29.6875], [55.46875, 27.734375], [54.6875, 26.5625], [55.46875, 25.390625], [53.90625, 25.0], [52.734375, 25.390625], [51.5625, 25.0], [50.390625, 24.21875], [48.828125, 23.828125], [47.65625, 24.21875], [46.484375, 23.4375], [45.3125, 24.21875], [46.09375, 26.953125], [44.921875, 27.34375], [43.359375, 26.171875], [41.40625, 24.609375], [40.234375, 23.046875]]
    p = PolygonObjectDetectionEvalItem(raw_data=None)
    polygon = p._try_build_poly(points)
    assert polygon.is_valid


def test_OCR_matching_function():
    res1 = [{
            "id": "rSbk_pk1g-",
            "type": "rectangle",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "bbox",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }, {
            "id": "rSbk_pk1g-",
            "type": "labels",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "labels": ["Text"],
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }, {
            "id": "rSbk_pk1g-",
            "type": "textarea",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "text": ["oh no"],
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "transcription",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }
        ]

    obj1 = OCREvalItem(res1)
    obj2 = OCREvalItem(res1)

    assert obj1.compare(obj2) == 1


def test_OCR_matching_function_no_rectangle():
    res1 = [{
            "id": "rSbk_pk1g-",
            "type": "rectangle",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "bbox",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }, {
            "id": "rSbk_pk1g-",
            "type": "labels",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "labels": ["Text"],
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }, {
            "id": "rSbk_pk1g-",
            "type": "textarea",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "text": ["oh no"],
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "transcription",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }
        ]
    res2 = [ {
            "id": "rSbk_pk1g",
            "type": "labels",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "labels": ["Text"],
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }, {
            "id": "rSbk_pk1g",
            "type": "textarea",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "text": ["oh no"],
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "transcription",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }
        ]

    obj1 = OCREvalItem(res1)
    obj2 = OCREvalItem(res2)

    assert obj1.compare(obj2) == 0


def test_OCR_matching_function_not_matching_text():
    res1 = [{
            "id": "rSbk_pk1g-",
            "type": "rectangle",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "bbox",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }, {
            "id": "rSbk_pk1g-",
            "type": "labels",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "labels": ["Text"],
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }, {
            "id": "rSbk_pk1g-",
            "type": "textarea",
            "value": {
                "x": 35.273972602739725,
                "y": 6.481481481481482,
                "text": ["oh no"],
                "width": 37.157534246575345,
                "height": 17.12962962962963,
                "rotation": 0
            },
            "to_name": "image",
            "from_name": "transcription",
            "image_rotation": 0,
            "original_width": 584,
            "original_height": 216
        }
        ]
    res2 = [{
        "id": "rSbk_pk1g",
        "type": "rectangle",
        "value": {
            "x": 35.273972602739725,
            "y": 6.481481481481482,
            "width": 37.157534246575345,
            "height": 17.12962962962963,
            "rotation": 0
        },
        "to_name": "image",
        "from_name": "bbox",
        "image_rotation": 0,
        "original_width": 584,
        "original_height": 216
    }, {
        "id": "rSbk_pk1g",
        "type": "labels",
        "value": {
            "x": 35.273972602739725,
            "y": 6.481481481481482,
            "width": 37.157534246575345,
            "height": 17.12962962962963,
            "labels": ["Text"],
            "rotation": 0
        },
        "to_name": "image",
        "from_name": "label",
        "image_rotation": 0,
        "original_width": 584,
        "original_height": 216
    }, {
        "id": "rSbk_pk1g",
        "type": "textarea",
        "value": {
            "x": 35.273972602739725,
            "y": 6.481481481481482,
            "text": ["ayyes"],
            "width": 37.157534246575345,
            "height": 17.12962962962963,
            "rotation": 0
        },
        "to_name": "image",
        "from_name": "transcription",
        "image_rotation": 0,
        "original_width": 584,
        "original_height": 216
    }
    ]
    obj1 = OCREvalItem(res1)
    obj2 = OCREvalItem(res2)

    assert obj1.compare(obj2) == 0
