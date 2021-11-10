from evalme.matcher import Matcher

from evalme.metrics import get_agreement


def test_agreement_matrix():
    m = Matcher()
    m.load(r"./tests/test_data/test_bbox.json")

    matrix = m.agreement_matrix()
    assert matrix is not None
    assert isinstance(matrix, dict)
    assert 187 in matrix.keys()


def test_get_agreement():
    """
    Test get_agreement method to calculate agreement
    """
    item_old = [{"id": "bEJvtrlNWk", "type": "hypertextlabels", "value": {"end": "/text()[1]",
                                                                          "text": "he other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with ",
                                                                          "start": "/text()[1]", "endOffset": 151,
                                                                          "htmllabels": ["Title"], "startOffset": 8,
                                                                          "globalOffsets": {"end": 151, "start": 8}},
                 "origin": "manual", "to_name": "text", "from_name": "ner"}]
    item_new = [{"id": "j_TEwQ0aZc", "type": "hypertextlabels", "value": {"end": "/text()[1]",
                                                                          "text": "One of the other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with me.",
                                                                          "start": "/text()[1]", "endOffset": 154,
                                                                          "startOffset": 0,
                                                                          "globalOffsets": {"end": 154, "start": 0},
                                                                          "hypertextlabels": ["Title"]},
                 "origin": "manual", "to_name": "text", "from_name": "ner"}]

    t1 = get_agreement(item_old, item_new)
    assert t1 == 0
    t2 = get_agreement(item_new, item_old)
    assert t2 == 0


def test_get_agreement_per_label():
    """
    Test get_agreement method to calculate agreement per label
    """
    item_old = [{"id": "bEJvtrlNWk", "type": "hypertextlabels", "value": {"end": "/text()[1]",
                                                                          "text": "he other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with ",
                                                                          "start": "/text()[1]", "endOffset": 151,
                                                                          "htmllabels": ["Title"], "startOffset": 8,
                                                                          "globalOffsets": {"end": 151, "start": 8}},
                 "origin": "manual", "to_name": "text", "from_name": "ner"}]
    item_new = [{"id": "j_TEwQ0aZc", "type": "hypertextlabels", "value": {"end": "/text()[1]",
                                                                          "text": "One of the other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked. They are right, as this is exactly what happened with me.",
                                                                          "start": "/text()[1]", "endOffset": 154,
                                                                          "startOffset": 0,
                                                                          "globalOffsets": {"end": 154, "start": 0},
                                                                          "hypertextlabels": ["Title"]},
                 "origin": "manual", "to_name": "text", "from_name": "ner"}]

    t1 = get_agreement(item_old, item_new, per_label=True)
    assert t1[1] == {'No-label': 0.0, 'Title': 0.0}
    t2 = get_agreement(item_new, item_old, per_label=True)
    assert t2[1] == {'No-label': 0.0, 'Title': 0.0}