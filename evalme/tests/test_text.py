import pytest
from evalme.text.text import HTMLTagsEvalItem, TaxonomyEvalItem, intersection_taxonomy, path_match_taxonomy


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
                            "Bacteria"
                        ],
                    ]
                },
                "to_name": "text",
                "from_name": "taxonomy"
            }
        ]]
    assert intersection_taxonomy(test_data[0], test_data[1]) == 1
    assert intersection_taxonomy(test_data[1], test_data[0]) == 1


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
    assert intersection_taxonomy(test_data[0], test_data[1], per_label=True) == {}
    assert intersection_taxonomy(test_data[1], test_data[0], per_label=True) == {}


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
    assert intersection_taxonomy(test_data[0], test_data[1], per_label=True) == {}
    assert intersection_taxonomy(test_data[1], test_data[0], per_label=True) == {}


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


def test_taxonomy_tree_with_parents_2():
    """
    Test for full tree with sparse tree
    """
    pred = path_match_taxonomy(tree1, tree2)
    assert pred == 1
    pred_vice = path_match_taxonomy(tree2, tree1)
    assert pred_vice == 0.5


def test_taxonomy_tree_with_parents_per_label():
    """
    Test for full tree with sparse tree per label
    """
    pred_label = intersection_taxonomy(tree1, tree2, label_config=label_config, per_label=True)
    assert pred_label == {"AAB": 1, "AAC": 1, "AC": 1, "BA": 1, "BC": 1, "CA": 1, "CC": 1}
    pred_label_vice = intersection_taxonomy(tree2, tree1, label_config=label_config, per_label=True)
    assert pred_label_vice == {"AAB": 1, "AAC": 1, "AC": 1,
                               "BA": 1, "BC": 1,
                               "CA": 1, "CC": 1}


def test_taxonomy_tree_with_parents_per_label_2():
    """
    Test for full tree with sparse tree per label
    """
    pred_label = path_match_taxonomy(tree1, tree2, per_label=True)
    assert pred_label == {}
    pred_label_vice = path_match_taxonomy(tree2, tree1, per_label=True)
    assert pred_label_vice == {}


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
    assert pred_label_vice == {}


first_leaf_tree = [{'value': {
        "taxonomy": [
            [
                "A",
                "AA"
            ],
        ]
    }}]

second_leaf_tree = [{'value': {
        "taxonomy": [
            [
                "C",
                "CA"
            ],
        ]
    }}]


def test_taxonomy_one_leaf_tree():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config)
    assert pred == 0
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config)
    assert pred_vice == 0


def test_taxonomy_one_leaf_tree_2():
    """
    Test for full tree with empty tree
    """
    pred = path_match_taxonomy(first_leaf_tree, second_leaf_tree)
    assert pred == 0
    pred_vice = path_match_taxonomy(second_leaf_tree, first_leaf_tree)
    assert pred_vice == 0


def test_taxonomy_one_leaf_tree_per_label():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config, per_label=True)
    assert pred == {}
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config, per_label=True)
    assert pred_vice == {}


label_config_with_leaf = r"""<View>
  <Text name="text" value="$text"/>
  <Taxonomy name="taxonomy" toName="text">
    <Choice value="A">
      <Choice value="AA"/>
    </Choice>
    <Choice value="B">
      <Choice value="BA">
        <Choice value="AAA"/>
      </Choice>
    </Choice>
    <Choice value="C">
      <Choice value="CA"/>
      <Choice value="CB"/>
      <Choice value="CC"/>
    </Choice>
    </Taxonomy>
</View>"""


def test_taxonomy_one_leaf_tree_with_leaf():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config_with_leaf)
    assert pred == 0
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config_with_leaf)
    assert pred_vice == 0


def test_taxonomy_one_leaf_tree_with_leaf_per_label():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config_with_leaf, per_label=True)
    assert pred == {}
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config_with_leaf, per_label=True)
    assert pred_vice == {}


label_config_subview = """
<View>
<Header value="This is test header" size="5" />
<Text name="url" value="$url" />
<View style="display: flex">
<View style="margin-left: auto">
<HyperText name="ht-1" value="$original_url_html"></HyperText>
</View>
</View>
<View>
<Taxonomy name="final_class" toName="extracted_content">
<Choice value="A" />
<Choice value="B" >
<Choice value="B_A" />
<Choice value="B_B" />
<Choice value="B_C" />
<Choice value="B_D" />
</Choice>
<Choice value="C" >
<Choice value="C_A" />
<Choice value="C_B" />
<Choice value="C_C" />
<Choice value="C_D" />
<Choice value="C_E" />
<Choice value="C_F" />
<Choice value="C_G" />
<Choice value="C_H" />
<Choice value="C_I" />
<Choice value="C_J" />
<Choice value="C_K" />
<Choice value="C_L" />
<Choice value="C_M" />
</Choice>
<Choice value="D" >
<Choice value="D_A" />
<Choice value="D_B" />
<Choice value="D_C" />
<Choice value="D_D" />
<Choice value="D_E" />
</Choice>
</Taxonomy>
</View>
</View>
"""

tree_subview_1 = [{'value': {
        "taxonomy": [
            [
                "B",
                "B_A"
            ],
        ]
    }}]
tree_subview_2 = [{'value': {
        "taxonomy": [
            [
                "B"
            ],
        ]
    }}]

def test_taxonomy_nested_label_config():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(tree_subview_1, tree_subview_2, label_config=label_config_subview)
    assert pred == 1
    pred_vice = intersection_taxonomy(tree_subview_2, tree_subview_1, label_config=label_config_subview)
    assert pred_vice == 0.25


def test_taxonomy_nested_label_config_2():
    """
    Test for full tree with empty tree
    """
    pred = path_match_taxonomy(tree_subview_1, tree_subview_2)
    assert pred == 0.5
    pred_vice = path_match_taxonomy(tree_subview_2, tree_subview_1)
    assert pred_vice == 1


def test_htmltags_migration():
    """
    Test html tags migration
    """
    item_old = [{"id": "bEJvtrlNWk", "type": "hypertextlabels", "value": {"end": "/text()[1]", "text": "he other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with ", "start": "/text()[1]", "endOffset": 151, "htmllabels": ["Title"], "startOffset": 8, "globalOffsets": {"end": 151, "start": 8}}, "origin": "manual", "to_name": "text", "from_name": "ner"}]
    item_new = [{"id": "j_TEwQ0aZc", "type": "hypertextlabels", "value": {"end": "/text()[1]", "text": "One of the other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with me.", "start": "/text()[1]", "endOffset": 154, "startOffset": 0, "globalOffsets": {"end": 154, "start": 0}, "hypertextlabels": ["Title"]}, "origin": "manual", "to_name": "text", "from_name": "ner"}]
    html_tags1 = HTMLTagsEvalItem(raw_data=item_old, shape_key="hypertextlabels")
    html_tags2 = HTMLTagsEvalItem(raw_data=item_new, shape_key="hypertextlabels")
    assert html_tags1.intersection(html_tags2) > 0.9
    assert html_tags2.intersection(html_tags1) > 0.9


def test_htmltags_migration_per_label():
    """
    Test html tags migration per label
    """
    item_old = [{"id": "bEJvtrlNWk", "type": "hypertextlabels", "value": {"end": "/text()[1]", "text": "he other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with ", "start": "/text()[1]", "endOffset": 151, "htmllabels": ["Title"], "startOffset": 8, "globalOffsets": {"end": 151, "start": 8}}, "origin": "manual", "to_name": "text", "from_name": "ner"}]
    item_new = [{"id": "j_TEwQ0aZc", "type": "hypertextlabels", "value": {"end": "/text()[1]", "text": "One of the other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with me.", "start": "/text()[1]", "endOffset": 154, "startOffset": 0, "globalOffsets": {"end": 154, "start": 0}, "hypertextlabels": ["Title"]}, "origin": "manual", "to_name": "text", "from_name": "ner"}]
    html_tags1 = HTMLTagsEvalItem(raw_data=item_old, shape_key="hypertextlabels")
    html_tags2 = HTMLTagsEvalItem(raw_data=item_new, shape_key="hypertextlabels")
    assert html_tags1.intersection(html_tags2, per_label=True) == {'Title': 0.9285714285714286}
    assert html_tags2.intersection(html_tags1, per_label=True) == {'Title': 0.9285714285714286}