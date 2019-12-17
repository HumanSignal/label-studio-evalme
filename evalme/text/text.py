from functools import partial

from evalme.eval_item import EvalItem


class HTMLTagsEvalItem(EvalItem):
    SHAPE_KEY = 'htmllabels'

    def _intersect(self, x, y):
        if x[self.SHAPE_KEY] != y[self.SHAPE_KEY]:
            return 0
        s1, e1 = x['start'], x['end']
        s2, e2 = y['start'], y['end']
        if s1 != s2 or e1 != e2:
            return 0
        s1, e1 = x['startOffset'], x['endOffset']
        s2, e2 = y['startOffset'], y['endOffset']
        if s2 > e1 or s1 > e2:
            return 0
        return min(e1, e2) - max(s1, s2)

    def intersection(self, item):
        item = _as_html_tags_eval_item(item)
        someone_is_empty = self.empty ^ item.empty
        if someone_is_empty:
            return 0
        if self.empty and item.empty:
            return 1
        total_len, total_intersect_len = 0, 0
        gt_values = self.get_values()
        for pred_value in item.get_values_iter():
            total_len += pred_value['endOffset'] - pred_value['startOffset']
            total_intersect_len += max(map(partial(self._intersect, y=pred_value), gt_values))
        return total_intersect_len / max(total_len, 1)


def _as_html_tags_eval_item(item):
    if not isinstance(item, HTMLTagsEvalItem):
        return HTMLTagsEvalItem(item)
    return item


def intersection_html_tagging(item_gt, item_pred):
    return _as_html_tags_eval_item(item_gt).intersection(item_pred)
