import textdistance

from itertools import zip_longest


_text_comparators = {}


def get_text_comparator(algorithm, qval):
    if algorithm is None and qval is None:
        # the default comparator fallbacks to simple text/labels equality
        return None
    comparator_key = (algorithm, qval)
    if comparator_key not in _text_comparators:
        _text_comparators[comparator_key] = getattr(textdistance, algorithm)(qval=qval)
    return _text_comparators[comparator_key].normalized_similarity


def texts_similarity(x, y, f=None):
    if not f:
        # the default comparator fallbacks to simple text/labels equality
        return x == y
    scores = []
    for xi, yi in zip_longest(x, y):
        if xi is None or yi is None:
            scores.append(0)
        else:
            scores.append(f(xi, yi))
    mean_score = sum(scores) / max(len(scores), 1)
    return mean_score