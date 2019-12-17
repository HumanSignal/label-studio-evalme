from evalme.eval_item import EvalItem


class HyperTextEvalItem(EvalItem):
    SHAPE_KEY = 'xpathlabels'

    def intersection(self, item):
        for value in self.get_values_iter():
