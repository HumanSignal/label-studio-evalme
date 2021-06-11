from evalme.matcher import Matcher

def test_agreement_matrix():
    m = Matcher()
    m.load(r"./test_data/test_bbox.json")

    matrix = m.agreement_matrix()
    assert matrix is not None
    assert isinstance(matrix, dict)
    assert 187 in matrix.keys()
