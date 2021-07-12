from evalme.metrics import Metrics

from evalme.image.object_detection import iou_polygons

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


def test_get_default_metric_for_name_tag_happy_path():
    result = Metrics.get_default_metric_for_name_tag('test', 'Test')
    assert result.description == 1


def test_get_default_metric_for_name_tag_no_metric_for_name():
    result = Metrics.get_default_metric_for_name_tag('Notfound', 'Notfound')
    assert result is None


def test_get_default_metric_for_name_tag_no_metric_for_name_but_for_tag():
    result = Metrics.get_default_metric_for_name_tag('test3', 'Notfound')
    assert result.description == 3


def test_get_default_metric_for_name_tag_happy_path():
    result = Metrics.get_default_metric_for_name_tag(tag='Notfound', name='Test4')
    assert result.description == 4


def test_clipped_project_dict():
    project = dict()
    project['id'] = 100
    res_1 = [{"id": "PMLDJIsBFp", "type": "polygonlabels", "value": {"points": [[25.466666666666665, 16.533333333333335], [18.666666666666668, 29.733333333333334], [17.2, 38], [29.866666666666667, 47.46666666666667], [37.46666666666667, 50], [58.8, 51.06666666666667], [72.13333333333334, 50.93333333333333], [75.2, 20.666666666666668], [45.46666666666667, 11.333333333333334]], "polygonlabels": ["Airplane"]}, "to_name": "image", "from_name": "label", "image_rotation": 0, "original_width": 1080, "original_height": 1080}, {"id": "Xr1N7057ag", "type": "polygonlabels", "value": {"points": [[16.666666666666668, 60.666666666666664], [50.8, 61.6], [78.26666666666667, 71.06666666666666], [80.53333333333333, 73.33333333333333], [78.4, 91.86666666666666], [22.266666666666666, 69.73333333333333]], "polygonlabels": ["Car"]}, "to_name": "image", "from_name": "label", "image_rotation": 0, "original_width": 1080, "original_height": 1080}]
    result = Metrics.apply(project, res_1, res_1)
    assert result == 1