from evalme.eval_item import EvalItem


class ClassificationEvalItem(EvalItem):

    SHAPE_KEY = 'undefined'

    def exact_match(self, item):
        if self.empty and item.empty:
            return 1
        if self.empty ^ item.empty:
            return 0
        if len(self) != len(item):
            return 0
        for x, y in zip(self.get_values_iter(), item.get_values_iter()):
            if x[self.SHAPE_KEY] != y[self.SHAPE_KEY]:
                return 0
        return 1


class ChoicesEvalItem(ClassificationEvalItem):
    SHAPE_KEY = 'choices'


class PairwiseEvalItem(ClassificationEvalItem):
    SHAPE_KEY = 'pairwise'
