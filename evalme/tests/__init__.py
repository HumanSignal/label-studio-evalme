from evalme.text.text import datetime_match, numbers_match, intersection_text_tagging, intersection_textarea_tagging, match_textareas, intersection_text_tagging
from evalme.classification import ClassificationEvalItem, ChoicesEvalItem, naive, exact_matching_choices
from evalme.metrics import Metrics
from evalme.image.object_detection import iou_polygons, iou_bboxes_textarea, iou_polygons_textarea


from functools import partial

Metrics.register(
    name="datetime naive",
    form='empty_form',
    tag='datetime',
    func=datetime_match,
    desc='Basic matching function for datetime',
    is_default=True
)


Metrics.register(
    name="number simple",
    form='empty_form',
    tag='number',
    func=numbers_match,
    desc='Exact matching function for number',
    is_default=True
)

Metrics.register(
    name='1d_region_intersection_threshold',
    form='iou_threshold',
    tag='Labels',
    func=intersection_text_tagging,
    desc='Percentage of matched regions by IOU w.r.t threshold'
)

Metrics.register(
    name='Test',
    form='',
    tag='Test',
    func='',
    desc=1
)

Metrics.register(
    name='Test2',
    form='',
    tag='Test, Test2',
    func='',
    desc=2
)

Metrics.register(
    name='Test3',
    form='',
    tag='Test3',
    func='',
    desc=3
)

Metrics.register(
    name='Test4',
    form='',
    tag='all',
    func='',
    desc=4
)

Metrics.register(
    name='iou_polygons',
    form='empty_form',
    tag='PolygonLabels',
    func=iou_polygons,
    desc='IOU for polygons'
)

Metrics.register(
    name='1d_region_intersection_threshold',
    form='iou_threshold',
    tag='Labels',
    func=intersection_text_tagging,
    desc='Percentage of matched regions by IOU w.r.t threshold'
)

Metrics.register(
    name='edit_distance',
    form='edit_distance',
    tag='TextArea',
    func=match_textareas,
    desc='Text edit distance'
)

Metrics.register(
    name='edit_distance_per_span',
    form='edit_distance',
    tag='TextArea[per_region=span]',
    func=partial(intersection_textarea_tagging, shape_key='text'),
    desc='Text edit distance per span region'
)

Metrics.register(
    name='edit_distance_per_span',
    form='iou_threshold',
    tag='TextArea[per_region=span]',
    func=partial(intersection_textarea_tagging, shape_key='text'),
    desc='Text edit distance per span region, with percentage of matched spans by IOU w.r.t threshold'
)

Metrics.register(
    name='edit_distance_per_hyperspan',
    form='iou_threshold',
    tag='TextArea[per_region=hyperspan]',
    func=partial(intersection_textarea_tagging, shape_key='text'),
    desc='Text edit distance per hypertext span region, with percentage of matched spans by IOU w.r.t threshold'
)

Metrics.register(
    name='edit_distance_per_bbox',
    form='edit_distance',
    tag='TextArea[per_region=bbox]',
    func=partial(iou_bboxes_textarea, shape_key='text'),
    desc='Text edit distance per bbox region'
)

Metrics.register(
    name='edit_distance_per_polygon',
    form='edit_distance',
    tag='TextArea[per_region=poly]',
    func=partial(iou_polygons_textarea, shape_key='text'),
    desc='Text edit distance per polygon region'
)