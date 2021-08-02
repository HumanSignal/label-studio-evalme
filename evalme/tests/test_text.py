import pytest
from evalme.text.text import HTMLTagsEvalItem, TaxonomyEvalItem, intersection_taxonomy


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


def test_hypertext_match():
    test_data = [{
        "text": "block1 <b><i>  block2 </i></b>",
        "htmllabels": ["Light"],
    },
        {
            "text": "block2 </i></b>",
            "htmllabels": ["Light"],
        }]
    obj = HTMLTagsEvalItem(raw_data=test_data)
    assert obj.spans_iou(test_data[0], test_data[1]) == 0.5
    assert obj.spans_iou(test_data[1], test_data[0]) == 0.5


def test_taxonomy_match():
    test_data = [[
        {
            "id": "Xg1qPZoLf_",
            "type": "taxonomy",
            "value": {
                "taxonomy": [
                    [
                        "Bacteria"
                    ],
                    [
                        "Eukarya"
                    ]
                ]
            },
            "to_name": "text",
            "from_name": "taxonomy"
        }
    ],
        [
            {
                "id": "Xg1qPZoLf_",
                "type": "taxonomy",
                "value": {
                    "taxonomy": [
                        [
                            "Archaea"
                        ],
                        [
                            "Bacteria"
                        ]
                    ]
                },
                "to_name": "text",
                "from_name": "taxonomy"
            }
        ]]
    assert intersection_taxonomy(test_data[0], test_data[1]) == 0.5
    assert intersection_taxonomy(test_data[1], test_data[0]) == 0.5


def test_taxonomy_match_perlabel():
    test_data = [[
        {
            "id": "Xg1qPZoLf_",
            "type": "taxonomy",
            "value": {
                "taxonomy": [
                    [
                        "Bacteria"
                    ],
                    [
                        "Eukarya"
                    ]
                ]
            },
            "to_name": "text",
            "from_name": "taxonomy"
        }
    ],
        [
            {
                "id": "Xg1qPZoLf_",
                "type": "taxonomy",
                "value": {
                    "taxonomy": [
                        [
                            "Archaea"
                        ],
                        [
                            "Bacteria"
                        ]
                    ]
                },
                "to_name": "text",
                "from_name": "taxonomy"
            }
        ]]
    assert intersection_taxonomy(test_data[0], test_data[1], per_label=True) == {"['Bacteria']": 1, "['Eukarya']": 0, "['Archaea']": 0}
    assert intersection_taxonomy(test_data[1], test_data[0], per_label=True) == {"['Bacteria']": 1, "['Eukarya']": 0, "['Archaea']": 0}


def test_taxonomy_doesn_match():
    test_data = [[
        {
            "id": "Xg1qPZoLf_",
            "type": "taxonomy",
            "value": {
                "taxonomy": [
                    [
                        "Bacteria"
                    ],
                    [
                        "Eukarya"
                    ]
                ]
            },
            "to_name": "text",
            "from_name": "taxonomy"
        }
    ],
        [
            {
                "id": "Xg1qPZoLf_",
                "type": "taxonomy",
                "value": {
                    "taxonomy": [
                        [
                            "Archaea"
                        ],
                        [
                            "Bacteria1"
                        ]
                    ]
                },
                "to_name": "text",
                "from_name": "taxonomy"
            }
        ]]
    assert intersection_taxonomy(test_data[0], test_data[1]) == 0.0
    assert intersection_taxonomy(test_data[1], test_data[0]) == 0.0


def test_taxonomy_doesn_match_perlabel():
    test_data = [[
        {
            "id": "Xg1qPZoLf_",
            "type": "taxonomy",
            "value": {
                "taxonomy": [
                    [
                        "Bacteria"
                    ],
                    [
                        "Eukarya"
                    ]
                ]
            },
            "to_name": "text",
            "from_name": "taxonomy"
        }
    ],
        [
            {
                "id": "Xg1qPZoLf_",
                "type": "taxonomy",
                "value": {
                    "taxonomy": [
                        [
                            "Archaea"
                        ],
                    ]
                },
                "to_name": "text",
                "from_name": "taxonomy"
            }
        ]]
    assert intersection_taxonomy(test_data[0], test_data[1], per_label=True) == {"['Bacteria']": 0, "['Eukarya']": 0, "['Archaea']": 0}
    assert intersection_taxonomy(test_data[1], test_data[0], per_label=True) == {"['Bacteria']": 0, "['Eukarya']": 0, "['Archaea']": 0}

