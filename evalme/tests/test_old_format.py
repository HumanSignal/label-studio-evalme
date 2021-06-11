from evalme.matcher import Matcher


def test_old_format_agreement_matrix():
    m = Matcher(new_format=False)
    m.load(r"./tests/test_data/test_old_format.json")

    matrix = m.get_annotations_agreement()
    assert matrix is not None
    assert matrix > 0


def test_old_format_load():
    m = Matcher(new_format=False)
    m.load(r"./tests/test_data/test_old_format.json")
    assert m._new_format is False
    assert m._result_name == 'completions'


def test_new_format_load():
    m = Matcher(new_format=False)
    m.load(r"./tests/test_data/test_bbox.json")
    assert m._new_format is True
    assert m._result_name == 'annotations'
