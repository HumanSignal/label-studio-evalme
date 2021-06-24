import pytest

from evalme.classification import ClassificationEvalItem, ChoicesEvalItem, naive


@pytest.mark.ClassificationEvalItem
def test_not_matching():
    test_data = [[
        {
            "from_name": "labels",
            "id": "6rhBThcT1F",
            "image_rotation": 0,
            "original_height": 5852,
            "original_width": 3902,
            "to_name": "image",
            "type": "polygonlabels",
            "value": {
                "points": [
                    [
                        43.333333333333336,
                        31.822222222222223
                    ],
                    [
                        34.8,
                        40.977777777777774
                    ],
                    [
                        38.266666666666666,
                        56.62222222222222
                    ],
                    [
                        61.2,
                        56.53333333333333
                    ],
                    [
                        65.6,
                        74.57777777777778
                    ],
                    [
                        89.73333333333333,
                        74.57777777777778
                    ],
                    [
                        86.13333333333334,
                        39.55555555555556
                    ]
                ],
                "polygonlabels": [
                    "Clothing"
                ]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories"]
                }
            }
        ]]
    obj = ChoicesEvalItem(raw_data=test_data[0])
    obj1 = ChoicesEvalItem(raw_data=test_data[1])
    assert obj1.exact_match(obj) == 0
    assert obj.exact_match(obj1) == 0


def test_not_matching_per_label():
    test_data = [[
        {
            "from_name": "labels",
            "id": "6rhBThcT1F",
            "image_rotation": 0,
            "original_height": 5852,
            "original_width": 3902,
            "to_name": "image",
            "type": "polygonlabels",
            "value": {
                "points": [
                    [
                        43.333333333333336,
                        31.822222222222223
                    ],
                    [
                        34.8,
                        40.977777777777774
                    ],
                    [
                        38.266666666666666,
                        56.62222222222222
                    ],
                    [
                        61.2,
                        56.53333333333333
                    ],
                    [
                        65.6,
                        74.57777777777778
                    ],
                    [
                        89.73333333333333,
                        74.57777777777778
                    ],
                    [
                        86.13333333333334,
                        39.55555555555556
                    ]
                ],
                "polygonlabels": [
                    "Clothing"
                ]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories"]
                }
            }
        ]]
    obj = ChoicesEvalItem(raw_data=test_data[0])
    obj1 = ChoicesEvalItem(raw_data=test_data[1])
    assert obj1.exact_match(obj, per_label=True) == {'Error': 0}
    assert obj.exact_match(obj1, per_label=True) == {'Error': 0}


def test_matching_type():
    test_data = [[
        {
            "from_name": "labels",
            "to_name": "image",
            "type": "choices",
            "value": {
                "choices": ["Accessories"]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories"]
                }
            }
        ]]
    obj = ChoicesEvalItem(raw_data=test_data[0])
    obj1 = ChoicesEvalItem(raw_data=test_data[1])
    assert obj1.exact_match(obj) == 1
    assert obj.exact_match(obj1) == 1


def test_matching_type_per_label():
    test_data = [[
        {
            "from_name": "labels",
            "to_name": "image",
            "type": "choices",
            "value": {
                "choices": ["Accessories"]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories"]
                }
            }
        ]]
    obj = ChoicesEvalItem(raw_data=test_data[0])
    obj1 = ChoicesEvalItem(raw_data=test_data[1])
    assert obj1.exact_match(obj, per_label=True) == {"Accessories": 1}
    assert obj.exact_match(obj1, per_label=True) == {"Accessories": 1}


def test_naive_matching():
    test_data = [[
        {
            "from_name": "labels",
            "to_name": "image",
            "type": "choices",
            "value": {
                "choices": ["Accessories"]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories"]
                }
            }
        ]]
    assert naive(test_data[0], test_data[1]) == 1


def test_naive_matching_per_label():
    test_data = [[
        {
            "from_name": "labels",
            "to_name": "image",
            "type": "choices",
            "value": {
                "choices": ["Accessories", "1", "2"]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories", "1", "2"]
                }
            }
        ]]
    assert naive(test_data[0], test_data[1], per_label=True) == {"Accessories\\1\\2": 1}


def test_naive_not_matching():
    test_data = [[
        {
            "from_name": "labels",
            "to_name": "image",
            "type": "choices",
            "value": {
                "choices": ["Accessories1"]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories2"]
                }
            }
        ]]
    assert naive(test_data[0], test_data[1]) == 0


def test_naive_not_matching_per_label():
    test_data = [[
        {
            "from_name": "labels",
            "to_name": "image",
            "type": "choices",
            "value": {
                "choices": ["Accessories1"]
            }
        }
    ],
        [
            {
                "from_name": "labels",
                "to_name": "image",
                "type": "choices",
                "value": {
                    "choices": ["Accessories2"]
                }
            }
        ]]
    assert naive(test_data[0], test_data[1], per_label=True) == {"Accessories1": 0}
