import pytest
from evalme.text.text import HTMLTagsEvalItem

def test_not_matching():
    test_data = [{
        "text": "dann steht halt",
        "htmllabels": ["Light negative"],
        },
        {
            "text": "internet steht ein",
            "htmllabels": ["Light negative"],
        }]
    obj = HTMLTagsEvalItem(raw_data=test_data)
    assert obj.spans_iou(test_data[0], test_data[1]) == 0
    assert obj.spans_iou(test_data[1], test_data[0]) == 0
    
def test_half_match():
    test_data = [{
        "text": "first second ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "htmllabels": ["Light"],
        },
        {
            "text": "ABCDEFGHIJKLMNOPQRSTUVWXYZ first second",
            "htmllabels": ["Light"],
        }]
    obj = HTMLTagsEvalItem(raw_data=test_data)
    assert obj.spans_iou(test_data[0], test_data[1]) == 0.5
    assert obj.spans_iou(test_data[1], test_data[0]) == 0.5
    
def test_different_labels():
    test_data = [{
        "text": "first second ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "htmllabels": ["Light1"],
        },
        {
            "text": "ABCDEFGHIJKLMNOPQRSTUVWXYZ first second",
            "htmllabels": ["Light2"],
        }]
    obj = HTMLTagsEvalItem(raw_data=test_data)
    assert obj.spans_iou(test_data[0], test_data[1]) == 0
    assert obj.spans_iou(test_data[1], test_data[0]) == 0
    
def test_full_part_another():
    test_data = [{
        "text": "first second ABCDEFGHIJKLM",
        "htmllabels": ["Light"],
    },
        {
            "text": "ABCDEFGHIJKLM",
            "htmllabels": ["Light"],
        }]
    obj = HTMLTagsEvalItem(raw_data=test_data)
    assert obj.spans_iou(test_data[0], test_data[1]) == 0.5
    assert obj.spans_iou(test_data[1], test_data[0]) == 0.5