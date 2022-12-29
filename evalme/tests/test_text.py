import pytest
from evalme.text.text import HTMLTagsEvalItem, TaxonomyEvalItem, intersection_taxonomy, path_match_taxonomy


def test_not_matching():
    """
    HTMLTagsEvalItem test iou match
    :return:
    """
    test_data = [{
        "start": 0,
        "end": 10,
        "text": "test_data",
        "htmllabels": ["Light negative"],
        },
        {
            "start": 11,
            "end": 20,
            "text": "test_data",
            "htmllabels": ["Light negative"],
        }]
    obj = HTMLTagsEvalItem(raw_data=test_data)
    assert obj.spans_iou(test_data[0], test_data[1]) == 0
    assert obj.spans_iou(test_data[1], test_data[0]) == 0


def test_half_match():
    test_data = [{
        "start": 120,
        "end": 130,
        "startOffset": 121,
        "endOffset": 124,
        "htmllabels": ["Light"],
        "text": "test_data",
        },
        {
            "start": 120,
            "end": 130,
            "startOffset": 122,
            "endOffset": 125,
            "htmllabels": ["Light"],
            "text": "different_text",
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
        "start": 120,
        "end": 130,
        "startOffset": 118,
        "endOffset": 124,
        "htmllabels": ["Light"],
        "text": "1",
    },
        {
            "start": 120,
            "end": 130,
            "startOffset": 121,
            "endOffset": 124,
            "htmllabels": ["Light"],
            "text": "22222",
        }]
    obj = HTMLTagsEvalItem(raw_data=test_data)
    assert obj.spans_iou(test_data[0], test_data[1]) == 0.5
    assert obj.spans_iou(test_data[1], test_data[0]) == 0.5


def test_hypertext_match():
    test_data = [{
        "start": 120,
        "end": 130,
        "startOffset": 118,
        "endOffset": 124,
        "text": "block1 <b><i>  block2 </i></b>",
        "htmllabels": ["Light"],
    },
        {
            "start": 120,
            "end": 130,
            "startOffset": 121,
            "endOffset": 124,
            "text": "block1 <b><i>  block2 </i></b>",
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
    pred = intersection_taxonomy(tree1, tree2, label_config=label_config, control_name='taxonomy')
    assert pred == 0.7
    pred_vice = intersection_taxonomy(tree2, tree1, label_config=label_config, control_name='taxonomy')
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
    pred_label = intersection_taxonomy(tree1, tree2, label_config=label_config, per_label=True, control_name='taxonomy')
    assert pred_label == {"AAB": 1, "AAC": 1, "AC": 1, "BA": 1, "BC": 1, "CA": 1, "CC": 1}
    pred_label_vice = intersection_taxonomy(tree2, tree1, label_config=label_config, per_label=True, control_name='taxonomy')
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
    pred = intersection_taxonomy(tree1, empty_tree, label_config=label_config, control_name='taxonomy')
    assert pred == 0
    pred_vice = intersection_taxonomy(empty_tree, tree1, label_config=label_config, control_name='taxonomy')
    assert pred_vice == 0


def test_taxonomy_empty_tree_with_parents_per_label():
    """
    Test for full tree with empty tree per label
    """
    pred_label = intersection_taxonomy(tree1, empty_tree, label_config=label_config, per_label=True, control_name='taxonomy')
    assert pred_label == {}
    pred_label_vice = intersection_taxonomy(empty_tree, tree1, label_config=label_config, per_label=True, control_name='taxonomy')
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
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config,
                                 control_name='taxonomy')
    assert pred == 0
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config,
                                      control_name='taxonomy')
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
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config, per_label=True,
                                 control_name='taxonomy')
    assert pred == {}
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config, per_label=True,
                                      control_name='taxonomy')
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
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config_with_leaf,
                                 control_name='taxonomy')
    assert pred == 0
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config_with_leaf,
                                      control_name='taxonomy')
    assert pred_vice == 0


def test_taxonomy_one_leaf_tree_with_leaf_per_label():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(first_leaf_tree, second_leaf_tree, label_config=label_config_with_leaf,
                                 per_label=True, control_name='taxonomy')
    assert pred == {}
    pred_vice = intersection_taxonomy(second_leaf_tree, first_leaf_tree, label_config=label_config_with_leaf,
                                      per_label=True, control_name='taxonomy')
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
<Taxonomy name="taxonomy" toName="extracted_content">
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
    pred = intersection_taxonomy(tree_subview_1, tree_subview_2, label_config=label_config_subview, control_name='taxonomy')
    assert pred == 1
    pred_vice = intersection_taxonomy(tree_subview_2, tree_subview_1, label_config=label_config_subview, control_name='taxonomy')
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


label_config_with_several_tags = """
<View>
<Header value="This is test header" size="5" />
<Text name="url" value="$url" />
<View style="display: flex">
<View style="margin-left: auto">
<HyperText name="ht-1" value="$original_url_html"></HyperText>
<Taxonomy name="taxonomy1" toName="extracted_content">
<Choice value="A" />
<Choice value="B" />
</Taxonomy>
<Taxonomy name="taxonomy4" toName="extracted_content">
<Choice value="TEST3" />
</Taxonomy>
</View>
</View>
<View>
<Taxonomy name="taxonomy" toName="extracted_content">
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


def test_taxonomy_several_tags_in_several_views():
    """
    Test taxonomy with several tags in config in different views
    :return:
    """
    pred = intersection_taxonomy(tree_subview_1, tree_subview_2, label_config=label_config_with_several_tags,
                                 control_name='taxonomy')
    assert pred == 1
    pred_vice = intersection_taxonomy(tree_subview_2, tree_subview_1, label_config=label_config_with_several_tags,
                                      control_name='taxonomy')
    assert pred_vice == 0.25


label_config_with_several_tags2 = """
<View>
<Header value="This is test header" size="5" />
<Text name="url" value="$url" />
<View style="display: flex">
<View style="margin-left: auto">
<HyperText name="ht-1" value="$original_url_html"></HyperText>
</View>
</View>
<View>
<Taxonomy name="taxonomy" toName="extracted_content">
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
<Taxonomy name="taxonomy2" toName="extracted_content">
<Choice value="TEST1" />
</Taxonomy>
<Taxonomy name="taxonomy3" toName="extracted_content">
<Choice value="TEST2" />
</Taxonomy>
</View>
</View>
"""


def test_taxonomy_several_tags_in_one_view():
    """
    Test taxonomy with several tags in config in one view
    :return:
    """
    pred = intersection_taxonomy(tree_subview_1, tree_subview_2, label_config=label_config_with_several_tags,
                                 control_name='taxonomy')
    assert pred == 1
    pred_vice = intersection_taxonomy(tree_subview_2, tree_subview_1, label_config=label_config_with_several_tags,
                                      control_name='taxonomy')
    assert pred_vice == 0.25


extra_label_confg = """
<View>
  <Text name="text" value="$text"/>
  <Taxonomy name="taxonomy" toName="text" placeholder="Default Taxonomy">
    <Choice value="Archaea"/>
    <Choice value="Bacteria"/>
    <Choice value="Eukarya">
      <Choice value="Human"/>
      <Choice value="Oppossum"/>
      <Choice value="Extraterrestial"/>
    </Choice>
  </Taxonomy>
  <Header>
leafsOnly, showFullPath, maxUsages=1, placeholder    </Header>
  <Taxonomy name="tax" toName="text" leafsOnly="true" showFullPath="true" placeholder="Choose your fighter" maxUsages="1">
    <Choice value="Alliance">
      <Choice value="Human"/>
      <Choice value="Dwarf"/>
      <Choice value="Night Elf"/>
      <Choice value="Gnome"/>
      <Choice value="Draenei"/>
      <Choice value="Worgen"/>
      <Choice value="Pandaren"/>
    </Choice>
    <Choice value="Horde">
      <Choice value="Orc"/>
      <Choice value="Undead"/>
      <Choice value="Tauren"/>
      <Choice value="Troll"/>
      <Choice value="Blood Elf"/>
      <Choice value="Goblin"/>
      <Choice value="Pandaren"/>
    </Choice>
  </Taxonomy>
  <Taxonomy name="uniq" toName="text" showFullPath="true" maxUsages="1">
      <Choice value="a">
        <Choice value="aa"/>
        <Choice value="ab">
          <Choice value="aba">
            <Choice value="abaa"/>
          </Choice>
          <Choice value="abb">
            <Choice value="abba"/>
            <Choice value="abbb">
              <Choice value="abbbc"/>
              <Choice value="abbbc"/>
              <Choice value="abbbc"/>
              <Choice value="abbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
            </Choice>
            <Choice value="abbc"/>
          </Choice>
        </Choice>
      </Choice>
      <Choice value="b">
        <Choice value="ba"/>
      </Choice>
      <Choice value="a">
        <Choice value="aa"/>
        <Choice value="ab">
          <Choice value="aba">
            <Choice value="abaa"/>
          </Choice>
          <Choice value="abb">
            <Choice value="abba"/>
            <Choice value="abbb">
              <Choice value="abbbc"/>
              <Choice value="abbbc"/>
              <Choice value="abbbc"/>
              <Choice value="abbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
              <Choice value="abbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbcabbbc"/>
            </Choice>
            <Choice value="abbc"/>
          </Choice>
        </Choice>
      </Choice>
      <Choice value="b">
        <Choice value="ba"/>
      </Choice>
    </Taxonomy>
  <Header>
Theese labels should have hotkeys 1, 2; perRegion required Taxonomy    </Header>
  <Labels name="lbl" toName="text">
    <Label value="Creature"/>
    <Label value="Rock"/>
    <Label value="Another" background="#FFA39E"/>
  </Labels>
  <Taxonomy name="creature" toName="text" required="true" perRegion="true">
    <Choice value="Apodanthaceae">
      <Choice value="Apodanthes"/>
      <Choice value="Berlinianche"/>
      <Choice value="Pilostyles"/>
    </Choice>
    <Choice value="Aponogetonaceae">
      <Choice value="Aponogeton"/>
    </Choice>
    <Choice value="Equisetaceae">
      <Choice value="Allostelites"/>
      <Choice value="Equisetum"/>
      <Choice value="Hippochaete"/>
    </Choice>
    <Choice value="Metaxyaceae">
      <Choice value="Metaxya"/>
    </Choice>
    <Choice value="Meteoriaceae">
      <Choice value="Aerobryidium"/>
      <Choice value="Aerobryopsis"/>
      <Choice value="Aerobryum"/>
      <Choice value="Ancistrodes"/>
      <Choice value="Barbella"/>
      <Choice value="Barbellopsis"/>
      <Choice value="Chrysocladium"/>
      <Choice value="Cryptopapillaria"/>
      <Choice value="Dicladiella"/>
      <Choice value="Floribundaria"/>
      <Choice value="Looseria"/>
      <Choice value="Meteoridium"/>
      <Choice value="Meteoriopsis"/>
      <Choice value="Meteorium"/>
      <Choice value="Neodicladiella"/>
      <Choice value="Neonoguchia"/>
      <Choice value="Orthostichella"/>
      <Choice value="Papillaria"/>
      <Choice value="Pilotrichella"/>
      <Choice value="Pseudobarbella"/>
      <Choice value="Sinskea"/>
      <Choice value="Squamidium"/>
      <Choice value="Toloxis"/>
      <Choice value="Trachycladiella"/>
      <Choice value="Weymouthia"/>
      <Choice value="Zelometeorium"/>
    </Choice>
    <Choice value="Metteniusaceae">
      <Choice value="Metteniusa"/>
    </Choice>
    <Choice value="Metzgeriaceae">
      <Choice value="Apometzgeria"/>
      <Choice value="Austrometzgeria"/>
      <Choice value="Echinomitrion"/>
      <Choice value="Metzgeria"/>
      <Choice value="Steereella"/>
    </Choice>
    <Choice value="Porellaceae">
      <Choice value="Bellincinia"/>
      <Choice value="Macvicaria"/>
      <Choice value="Madotheca"/>
      <Choice value="Porella"/>
    </Choice>
    <Choice value="Rutenbergiaceae">
      <Choice value="Neorutenbergia"/>
      <Choice value="Rutenbergia"/>
    </Choice>
    <Choice value="Tofieldiaceae">
      <Choice value="Harperocallis"/>
      <Choice value="Pleea"/>
      <Choice value="Tofieldia"/>
      <Choice value="Triantha"/>
    </Choice>
    <Choice value="Torricelliaceae">
      <Choice value="Aralidium"/>
      <Choice value="Melanophylla"/>
    </Choice>
    <Choice value="Tovariaceae">
      <Choice value="Tovaria"/>
    </Choice>
    <Choice value="Trachypodaceae">
      <Choice value="Bryowijkia"/>
      <Choice value="Diaphanodon"/>
      <Choice value="Pseudospiridentopsis"/>
      <Choice value="Pseudotrachypus"/>
      <Choice value="Trachypodopsis"/>
      <Choice value="Trachypus"/>
    </Choice>
    <Choice value="Treubiaceae">
      <Choice value="Apotreubia"/>
      <Choice value="Treubia"/>
    </Choice>
  </Taxonomy>
</View>
"""


def test_taxonomy_extra_label_in_annotations():
    """
    Test taxonomy with several tags in config in one view
    :return:
    """
    r1 = [{'id': 'XtM3YzHXLw', 'type': 'taxonomy', 'value': {'taxonomy': [['Tauren']]}, 'origin': 'manual', 'to_name': 'text', 'from_name': 'tax'}]
    r2 = [{'id': 'Wl2JXU9zIi', 'type': 'taxonomy', 'value': {'taxonomy': [['Horde', 'Pandaren']]}, 'origin': 'manual', 'to_name': 'text', 'from_name': 'tax'}]
    pred = intersection_taxonomy(r1, r2, label_config=extra_label_confg,
                                 control_name='tax')
    assert pred == 0
    pred_vice = intersection_taxonomy(r1, r2, label_config=extra_label_confg,
                                      control_name='uniq')
    assert pred_vice == 0


tree_subview_with_new_label = [{'value': {
        "taxonomy": [
            [
                "B", "E"
            ],
        ]
    }}]


def test_taxonomy_nested_label_config_with_added_label():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(tree_subview_1, tree_subview_with_new_label, label_config=label_config_subview, control_name='taxonomy')
    assert pred == 0
    pred_vice = intersection_taxonomy(tree_subview_with_new_label, tree_subview_1, label_config=label_config_subview, control_name='taxonomy')
    assert pred_vice == 0.0

tree_subview_with_new_label1 = [{'value': {
        "taxonomy": [
            [
                "B"
            ],
            [
                "E"
            ]
        ]
    }}]


def test_taxonomy_nested_label_config_with_added_label():
    """
    Test for full tree with empty tree
    """
    pred = intersection_taxonomy(tree_subview_1, tree_subview_with_new_label1, label_config=label_config_subview, control_name='taxonomy')
    assert pred == 1.0
    pred_vice = intersection_taxonomy(tree_subview_with_new_label1, tree_subview_1, label_config=label_config_subview, control_name='taxonomy')
    assert pred_vice == 0.2
