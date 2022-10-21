from evalme.utils import texts_similarity


def test_texts_similarity():
    result = texts_similarity(x="1", y="1234", f=lambda x, y: int(x==y))
    assert result == 0.25