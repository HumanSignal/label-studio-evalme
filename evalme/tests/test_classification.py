import pytest

from evalme.classification import ClassificationEvalItem, ChoicesEvalItem, naive, exact_matching_choices

from evalme.metrics import Metrics


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
    assert Metrics.apply({}, test_data[0], test_data[1], metric_name='naive') == 1


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
    assert Metrics.apply({}, test_data[0], test_data[1], metric_name='naive', per_label=True) == {"Accessories\\1\\2": 1}


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
    assert Metrics.apply({}, test_data[0], test_data[1], metric_name='naive') == 0


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
    assert Metrics.apply({}, test_data[0], test_data[1], metric_name='naive', per_label=True) == {"Accessories1": 0, "Accessories2": 0}


def test_dynamic_choices():
    test_data1 = [{'value': {'choices': [['Products', 'Loan Payment Center']]}, 'id': 'edeDdGdNnb',
                   'from_name': 'dynamic_choices', 'to_name': 'text', 'type': 'choices', 'origin': 'manual'}]
    score = exact_matching_choices(test_data1,
                                   test_data1,
                                   {})
    score_per_label = exact_matching_choices(test_data1,
                                             test_data1,
                                             {},
                                             per_label=True)
    assert score == 1
    assert score_per_label == {'Products': 1, 'Loan Payment Center': 1}


def test_choices_diff_choices_groups():
    test_data1 = [{"id": "QAZRsnSniS", "type": "choices", "value": {"choices": ["Negative"]}, "origin": "manual",
                   "to_name": "text", "from_name": "sentiment1"}]
    test_data2 = [{"id": "p8If1f0cDV", "type": "choices", "value": {"choices": ["Negative2"]}, "origin": "manual",
                   "to_name": "text", "from_name": "sentiment2"}]
    score = exact_matching_choices(test_data1,
                                   test_data2,
                                   {})
    score_per_label = exact_matching_choices(test_data1,
                                             test_data2,
                                             {},
                                             per_label=True)
    assert score == 0
    assert score_per_label == {'Negative': 0}
    score = exact_matching_choices(test_data2,
                                   test_data1,
                                   {})
    score_per_label = exact_matching_choices(test_data2,
                                             test_data1,
                                             {},
                                             per_label=True)
    assert score == 0
    assert score_per_label == {'Negative2': 0}


def test_naive_for_htmllabels():
    result = [{"from_name":'header',"to_name":'text',"type":'hypertextlabels',
               "value":{"end":'/section/',"endOffset":1,"htmllabels":["data"],"start":'/section/',"startOffset":0,
                        "text":'1/1/2023'}}]
    assert naive(result, result) == 1
    assert naive(result, result, per_label=True) == {"data": 1}
