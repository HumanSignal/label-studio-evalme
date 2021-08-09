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

tree1 = [{"value": {
        "taxonomy": [
            [
                "A"
            ],
            [
                "B"
            ],
            [
                "C"
            ]
        ]
    }}]
tree2 = [{'value': {
        "taxonomy": [
            [
                "A",
                "AA"
            ],
            [
                "A",
                "AC"
            ],
            [
                "B",
                "BA"
            ],
            [
                "B",
                "BC"
            ],
            [
                "C",
                "CA"
            ],
            [
                "C",
                "CC"
            ]
        ]
    }}]
label_config = r"""<View>
  <Text name="text" value="$text"/>
  <Taxonomy name="taxonomy" toName="text">
    <Choice value="A">
      <Choice value="AA">
        <Choice value="AAB"/>
        <Choice value="AAC"/>
      </Choice>
      <Choice value="AB"/>
      <Choice value="AC"/>
    </Choice>
    <Choice value="B">
      <Choice value="BA"/>
      <Choice value="BB"/>
      <Choice value="BC"/>
    </Choice>
    <Choice value="C">
      <Choice value="CA"/>
      <Choice value="CB"/>
      <Choice value="CC"/>
    </Choice>
    </Taxonomy>
</View>"""


def test_taxonomy_tree_with_parents():
    """
    Test for full tree with sparse tree
    """
    pred = intersection_taxonomy(tree1, tree2, label_config=label_config)
    assert pred == 0.7
    pred_vice = intersection_taxonomy(tree2, tree1, label_config=label_config)
    assert pred_vice == 1


def test_taxonomy_tree_with_parents_per_label():
    """
    Test for full tree with sparse tree per label
    """
    pred_label = intersection_taxonomy(tree1, tree2, label_config=label_config, per_label=True)
    assert pred_label == {"AAB": 1, "AAC": 1, "AC": 1, "BA": 1, "BC": 1, "CA": 1, "CC": 1}
    pred_label_vice = intersection_taxonomy(tree2, tree1, label_config=label_config, per_label=True)
    assert pred_label_vice == {"AAB": 1, "AAC": 1, "AB":0, "AC": 1,
                               "BA": 1, "BB": 0, "BC": 1,
                               "CA": 1, "CB": 0, "CC": 1}

empty_tree = [{'value': {
        "taxonomy": [
        ]
    }}]

def test_taxonomy_empty_tree_with_parents():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(tree1, empty_tree, label_config=label_config)
    assert pred == 0
    pred_vice = intersection_taxonomy(empty_tree, tree1, label_config=label_config)
    assert pred_vice == 0


def test_taxonomy_empty_tree_with_parents_per_label():
    """
    Test for full tree with empty tree per label
    """
    pred_label = intersection_taxonomy(tree1, empty_tree, label_config=label_config, per_label=True)
    assert pred_label == {}
    pred_label_vice = intersection_taxonomy(empty_tree, tree1, label_config=label_config, per_label=True)
    assert pred_label_vice == {"AAB": 0, "AAC": 0, "AB": 0, "AC": 0,
                               "BA": 0, "BB": 0, "BC": 0,
                               "CA": 0, "CB": 0, "CC": 0}
