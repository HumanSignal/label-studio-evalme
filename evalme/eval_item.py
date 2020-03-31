from operator import itemgetter


class EvalItem(object):
    """
    Generic class that contains all info about evaluation item
    """

    SHAPE_KEY = None

    def __init__(self, raw_data, shape_key=None):
        self._raw_data = raw_data
        self._shape_key = shape_key or self.SHAPE_KEY
        if not self._shape_key:
            raise ValueError('Shape key is undefined')

    @property
    def raw_data(self):
        return self._raw_data

    def get_values_iter(self):
        return map(itemgetter('value'), self._raw_data)

    def get_values(self):
        return list(self.get_values_iter())

    @property
    def empty(self):
        return len(self._raw_data) == 0

    def __len__(self):
        return len(self._raw_data)
