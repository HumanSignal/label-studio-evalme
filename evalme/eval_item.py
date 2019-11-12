from operator import itemgetter


class EvalItem(object):
    """
    Generic class that contains all info about evaluation item
    """
    def __init__(self, raw_data):
        self._raw_data = raw_data

    @property
    def raw_data(self):
        return self._raw_data

    def get_values_iter(self):
        return map(itemgetter('value'), self._raw_data)

    def get_values(self):
        return list(self.get_values_iter())
