from evalme.matcher import Matcher

def test_old_format_agreement_matrix():
    m = Matcher(new_format=False)
    m.load(r"test_data\test_old_format.json")

    matrix = m.get_annotations_agreement()
    assert matrix is not None
    assert matrix > 0
