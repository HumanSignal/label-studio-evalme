import pytest
from evalme.text.text import intersection_text_tagging


def test_same_timeseries_labels():
    """
    HTMLTagsEvalItem test iou match
    :return:
    """
    test_data = [{"value":{
        "end": "2022-12-26 13:24:18",
        "start": "2022-12-26 12:33:37",
        "instant": False,
        "timeserieslabels": [
            "in bed"
        ]
    }},{"value": {
        "end": "2022-12-26 13:28:23",
        "start": "2022-12-26 12:33:37",
        "instant": False,
        "timeserieslabels": [
            "in bed"
        ]
    }}]

    assert round(intersection_text_tagging([test_data[0]], [test_data[1]], shape_key='timeserieslabels')) == 1
    assert round(intersection_text_tagging([test_data[1]], [test_data[0]], shape_key='timeserieslabels')) == 1
    feature_flags = {}
    feature_flags['ff_back_dev_2762_textarea_weights_30062022_short'] = True
    assert round(intersection_text_tagging([test_data[0]], [test_data[1]], shape_key='timeserieslabels', feature_flags=feature_flags)) == 1
    assert round(intersection_text_tagging([test_data[1]], [test_data[0]], shape_key='timeserieslabels', feature_flags=feature_flags)) == 1


def test_different_timeseries_labels():
    """
    HTMLTagsEvalItem test iou match
    :return:
    """
    test_data = [{"value":{
        "end": "2022-12-26 13:24:18",
        "start": "2022-12-26 12:33:37",
        "instant": False,
        "timeserieslabels": [
            "in bed1"
        ]
    }},{"value": {
        "end": "2022-12-26 13:28:23",
        "start": "2022-12-26 12:33:37",
        "instant": False,
        "timeserieslabels": [
            "in bed2"
        ]
    }}]

    assert round(intersection_text_tagging([test_data[0]], [test_data[1]], shape_key='timeserieslabels')) == 0
    assert round(intersection_text_tagging([test_data[1]], [test_data[0]], shape_key='timeserieslabels')) == 0
    feature_flags = {}
    feature_flags['ff_back_dev_2762_textarea_weights_30062022_short'] = True
    assert round(intersection_text_tagging([test_data[0]], [test_data[1]], shape_key='timeserieslabels', feature_flags=feature_flags)) == 0
    assert round(intersection_text_tagging([test_data[1]], [test_data[0]], shape_key='timeserieslabels', feature_flags=feature_flags)) == 0