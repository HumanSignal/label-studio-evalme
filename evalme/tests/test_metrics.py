from evalme.metrics import Metrics

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

def test_get_default_metric_for_name_tag_happy_path():
    result = Metrics.get_default_metric_for_name_tag('Test', 'Test')
    assert result.description == 1


def test_get_default_metric_for_name_tag_no_metric_for_name():
    result = Metrics.get_default_metric_for_name_tag('Notfound', 'Notfound')
    assert result is None

def test_get_default_metric_for_name_tag_no_metric_for_name_but_for_tag():
    result = Metrics.get_default_metric_for_name_tag('Notfound', 'Test3')
    assert result.description == 3