from evalme.image.object_detection import KeyPointsEvalItem, keypoints_distance, PolygonObjectDetectionEvalItem, \
    OCREvalItem, ocr_compare


def test_ocr_matching_function():
    """
    Compare OCR with same annotation
    """
    res1 = {"result": [{"id": "rSbk_pk1g-", "type": "rectangle",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "rotation": 0}, "to_name": "image", "from_name": "bbox",
                        "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g-", "type": "labels",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "labels": ["Text"], "rotation": 0}, "to_name": "image",
                        "from_name": "label", "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g-", "type": "textarea",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "text": ["oh no"],
                                  "width": 37.157534246575345, "height": 17.12962962962963, "rotation": 0},
                        "to_name": "image", "from_name": "transcription", "image_rotation": 0, "original_width": 584,
                        "original_height": 216}]}

    obj1 = OCREvalItem(res1)
    obj2 = OCREvalItem(res1)

    assert obj1.compare(obj2) == 1


def test_ocr_matching_function_no_rectangle():
    """
    1 Annotation doesn't have a Rectangle in result
    """
    res1 = {"result": [{"id": "rSbk_pk1g-", "type": "rectangle",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "rotation": 0}, "to_name": "image", "from_name": "bbox",
                        "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g-", "type": "labels",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "labels": ["Text"], "rotation": 0}, "to_name": "image",
                        "from_name": "label", "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g-", "type": "textarea",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "text": ["oh no"],
                                  "width": 37.157534246575345, "height": 17.12962962962963, "rotation": 0},
                        "to_name": "image", "from_name": "transcription", "image_rotation": 0, "original_width": 584,
                        "original_height": 216}]}
    res2 = {"result": [{"id": "rSbk_pk1g", "type": "labels",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "labels": ["Text"], "rotation": 0}, "to_name": "image",
                        "from_name": "label", "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g", "type": "textarea",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "text": ["oh no"],
                                  "width": 37.157534246575345, "height": 17.12962962962963, "rotation": 0},
                        "to_name": "image", "from_name": "transcription", "image_rotation": 0, "original_width": 584,
                        "original_height": 216}]}

    obj1 = OCREvalItem(res1)
    obj2 = OCREvalItem(res2)

    assert obj1.compare(obj2) == 0


def test_ocr_matching_function_not_matching_text():
    """
    OCR annotations with different text
    """
    res1 = {"result": [{"id": "rSbk_pk1g-", "type": "rectangle",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "rotation": 0}, "to_name": "image", "from_name": "bbox",
                        "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g-", "type": "labels",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "labels": ["Text"], "rotation": 0}, "to_name": "image",
                        "from_name": "label", "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g-", "type": "textarea",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "text": ["oh no"],
                                  "width": 37.157534246575345, "height": 17.12962962962963, "rotation": 0},
                        "to_name": "image", "from_name": "transcription", "image_rotation": 0, "original_width": 584,
                        "original_height": 216}]}
    res2 = {"result": [{"id": "rSbk_pk1g", "type": "rectangle",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "rotation": 0}, "to_name": "image", "from_name": "bbox",
                        "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g", "type": "labels",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "width": 37.157534246575345,
                                  "height": 17.12962962962963, "labels": ["Text"], "rotation": 0}, "to_name": "image",
                        "from_name": "label", "image_rotation": 0, "original_width": 584, "original_height": 216},
                       {"id": "rSbk_pk1g", "type": "textarea",
                        "value": {"x": 35.273972602739725, "y": 6.481481481481482, "text": ["ayyes"],
                                  "width": 37.157534246575345, "height": 17.12962962962963, "rotation": 0},
                        "to_name": "image", "from_name": "transcription", "image_rotation": 0, "original_width": 584,
                        "original_height": 216}]}
    obj1 = OCREvalItem(res1)
    obj2 = OCREvalItem(res2)

    assert obj1.compare(obj2) == 0


def test_ocr_simple_matching():
    """
    Simple OCR matching with annotations with all controls
    """
    from evalme.metrics import Metrics

    Metrics.register(
        name='OCR',
        form='empty_form',
        tag='all',
        func=ocr_compare,
        desc='OCR distance'
    )

    ann1 = {"result": [{'original_width': 768, 'original_height': 576, 'image_rotation': 0,
                        'value': {'x': 64.53333333333333, 'y': 59.502664298401434, 'width': 19.19999999999997,
                                  'height': 12.07815275310836, 'rotation': 0}, 'id': '9VXbGdgh0T', 'from_name': 'bbox',
                        'to_name': 'image', 'type': 'rectangle', 'origin': 'manual'},
                       {'original_width': 768, 'original_height': 576, 'image_rotation': 0,
                        'value': {'x': 64.53333333333333, 'y': 59.502664298401434, 'width': 19.19999999999997,
                                  'height': 12.07815275310836, 'rotation': 0, 'labels': ['Text']}, 'id': '9VXbGdgh0T',
                        'from_name': 'label', 'to_name': 'image', 'type': 'labels', 'origin': 'manual'},
                       {'original_width': 768, 'original_height': 576, 'image_rotation': 0,
                        'value': {'x': 64.53333333333333, 'y': 59.502664298401434, 'width': 19.19999999999997,
                                  'height': 12.07815275310836, 'rotation': 0, 'text': ['17-RX-RR']}, 'id': '9VXbGdgh0T',
                        'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea', 'origin': 'manual'}]}
    ann2 = {"result": [{'id': 'buquXLcKOL', 'type': 'rectangle',
                        'value': {'x': 63.6, 'y': 60.92362344582593, 'width': 20.666666666666668,
                                  'height': 10.8348134991119, 'rotation': 0}, 'origin': 'manual', 'to_name': 'image',
                        'from_name': 'bbox', 'image_rotation': 0, 'original_width': 768, 'original_height': 576},
                       {'id': 'buquXLcKOL', 'type': 'labels',
                        'value': {'x': 63.6, 'y': 60.92362344582593, 'width': 20.666666666666668,
                                  'height': 10.8348134991119, 'labels': ['Text'], 'rotation': 0}, 'origin': 'manual',
                        'to_name': 'image', 'from_name': 'label', 'image_rotation': 0, 'original_width': 768,
                        'original_height': 576},
                       {'id': 'buquXLcKOL', 'type': 'textarea',
                        'value': {'x': 63.6, 'y': 60.92362344582593, 'text': ['17-RX-RR'],
                                  'width': 20.666666666666668, 'height': 10.8348134991119,
                                  'rotation': 0}, 'origin': 'manual', 'to_name': 'image',
                        'from_name': 'transcription', 'image_rotation': 0,
                        'original_width': 768, 'original_height': 576}]}
    score = Metrics.apply({}, ann1, ann2, metric_name='OCR')
    assert score == 1.0


def test_ocr_matching_with_several_control_types():
    """
    Annotations with results with several control types
    :return:
    """
    from evalme.metrics import Metrics

    Metrics.register(
        name='OCR',
        form='empty_form',
        tag='all',
        func=ocr_compare,
        desc='OCR distance'
    )

    ann1 = {"result": [{"id": "PMAZ9d76PO", "type": "rectangle",
                        "value": {"x": 15, "y": 18.333333333333332, "width": 44.375, "height": 16.666666666666664,
                                  "rotation": 0}, "origin": "manual", "to_name": "image", "from_name": "bbox",
                        "image_rotation": 0, "original_width": 320, "original_height": 240},
                       {"id": "PMAZ9d76PO", "type": "labels",
                        "value": {"x": 15, "y": 18.333333333333332, "width": 44.375, "height": 16.666666666666664,
                                  "labels": ["Text"], "rotation": 0}, "origin": "manual", "to_name": "image",
                        "from_name": "label", "image_rotation": 0, "original_width": 320, "original_height": 240},
                       {"id": "PMAZ9d76PO", "type": "textarea",
                        "value": {"x": 15, "y": 18.333333333333332, "text": ["Text"], "width": 44.375,
                                  "height": 16.666666666666664, "rotation": 0}, "origin": "manual", "to_name": "image",
                        "from_name": "transcription", "image_rotation": 0, "original_width": 320,
                        "original_height": 240},
                       {"id": "IgTTgpaMvm", "type": "textarea", "value": {"text": ["Text"]}, "origin": "manual",
                        "to_name": "audio", "from_name": "transcription1"}]}
    ann2 = {"result": [{"id": "eB4V3hkedQ", "type": "rectangle",
                        "value": {"x": 14.0625, "y": 18.333333333333332, "width": 41.5625, "height": 17.083333333333332,
                                  "rotation": 0}, "origin": "manual", "to_name": "image", "from_name": "bbox",
                        "image_rotation": 0, "original_width": 320, "original_height": 240},
                       {"id": "eB4V3hkedQ", "type": "labels",
                        "value": {"x": 14.0625, "y": 18.333333333333332, "width": 41.5625, "height": 17.083333333333332,
                                  "labels": ["Text"], "rotation": 0}, "origin": "manual", "to_name": "image",
                        "from_name": "label", "image_rotation": 0, "original_width": 320, "original_height": 240},
                       {"id": "eB4V3hkedQ", "type": "textarea",
                        "value": {"x": 14.0625, "y": 18.333333333333332, "text": ["Text"], "width": 41.5625,
                                  "height": 17.083333333333332, "rotation": 0}, "origin": "manual", "to_name": "image",
                        "from_name": "transcription", "image_rotation": 0, "original_width": 320,
                        "original_height": 240},
                       {"id": "3Qx2-JNxjz", "type": "textarea", "value": {"text": ["Text"]}, "origin": "manual",
                        "to_name": "audio", "from_name": "transcription1"}]}
    score = Metrics.apply({}, ann1, ann2, metric_name='OCR')
    assert score == 0.5


def test_ocr_2_groups_of_regions_with_labels():
    """
    DEV-1721
    2 regions with same labels and 2 regions with different labels with Text
    :return:
    """
    result1 = {"result": [{"id": "Tp5yC-6hsd", "type": "rectangle",
                           "value": {"x": 26.666666666666632, "y": 30.933333333333323, "width": 12.933333333333314,
                                     "height": 14.933333333333312, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "bbox", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Tp5yC-6hsd", "type": "labels",
                           "value": {"x": 26.666666666666632, "y": 30.933333333333323,
                                     "width": 12.933333333333314,
                                     "height": 14.933333333333312, "labels": ["Text"],
                                     "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Tp5yC-6hsd", "type": "textarea",
                           "value": {"x": 26.666666666666632,
                                     "y": 30.933333333333323,
                                     "text": ["aaa bbb"],
                                     "width": 12.933333333333314,
                                     "height": 14.933333333333312,
                                     "rotation": 0},
                           "origin": "manual", "to_name": "image",
                           "from_name": "transcription",
                           "image_rotation": 0,
                           "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Iuq3q9qQXR", "type": "rectangle",
                           "value": {"x": 63.46666666666667, "y": 53.86666666666666, "width": 28.933333333333334,
                                     "height": 17.733333333333334, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "bbox", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Iuq3q9qQXR", "type": "labels",
                           "value": {"x": 63.46666666666667, "y": 53.86666666666666,
                                     "width": 28.933333333333334,
                                     "height": 17.733333333333334, "labels": ["Handwriting"],
                                     "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Iuq3q9qQXR", "type": "textarea",
                           "value": {"x": 63.46666666666667,
                                     "y": 53.86666666666666,
                                     "text": ["name"],
                                     "width": 28.933333333333334,
                                     "height": 17.733333333333334,
                                     "rotation": 0},
                           "origin": "manual", "to_name": "image",
                           "from_name": "transcription",
                           "image_rotation": 0,
                           "original_width": 1080,
                           "original_height": 1080}]}
    result2 = {"result": [{"id": "rWSy2ZUZm7", "type": "rectangle",
                           "value": {"x": 26.000000000000036, "y": 31.6, "width": 13.066666666666638,
                                     "height": 14.133333333333297, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "bbox", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "rWSy2ZUZm7", "type": "textarea",
                           "value": {"x": 26.000000000000036, "y": 31.6, "text": ["aaa bbb"],
                                     "width": 13.066666666666638,
                                     "height": 14.133333333333297, "rotation": 0},
                           "origin": "manual", "to_name": "image",
                           "from_name": "transcription", "image_rotation": 0,
                           "original_width": 1080, "original_height": 1080},
                          {"id": "rWSy2ZUZm7", "type": "labels",
                           "value": {"x": 26.000000000000036, "y": 31.6, "width": 13.066666666666638,
                                     "height": 14.133333333333297, "labels": ["Text"], "rotation": 0},
                           "origin": "manual", "to_name": "image", "from_name": "label", "image_rotation": 0,
                           "original_width": 1080, "original_height": 1080},
                          {"id": "ePS5El8SPd", "type": "rectangle",
                           "value": {"x": 62.93333333333333,
                                     "y": 55.2,
                                     "width": 28.799999999999997,
                                     "height": 15.6, "rotation": 0},
                           "origin": "manual", "to_name": "image",
                           "from_name": "bbox", "image_rotation": 0,
                           "original_width": 1080,
                           "original_height": 1080},
                          {"id": "ePS5El8SPd", "type": "labels",
                           "value": {"x": 62.93333333333333, "y": 55.2, "width": 28.799999999999997, "height": 15.6,
                                     "labels": ["Text"], "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 1080, "original_height": 1080},
                          {"id": "ePS5El8SPd", "type": "textarea",
                           "value": {"x": 62.93333333333333, "y": 55.2, "text": ["form"], "width": 28.799999999999997,
                                     "height": 15.6, "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "transcription", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080}]}
    o1 = OCREvalItem(result1)
    o2 = OCREvalItem(result2)
    score = o1.compare(o2)
    assert score == 0.5


def test_ocr_2_groups_of_regions_without_text():
    """
    DEV-1721
    2 regions with same labels and 2 regions with different labels (without Text)
    :return:
    """
    result1 = {"result": [{"id": "Tp5yC-6hsd", "type": "rectangle",
                           "value": {"x": 26.666666666666632, "y": 30.933333333333323, "width": 12.933333333333314,
                                     "height": 14.933333333333312, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "bbox", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Tp5yC-6hsd", "type": "labels",
                           "value": {"x": 26.666666666666632, "y": 30.933333333333323,
                                     "width": 12.933333333333314,
                                     "height": 14.933333333333312, "labels": ["Text"],
                                     "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Iuq3q9qQXR", "type": "rectangle",
                           "value": {"x": 63.46666666666667, "y": 53.86666666666666, "width": 28.933333333333334,
                                     "height": 17.733333333333334, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "bbox", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "Iuq3q9qQXR", "type": "labels",
                           "value": {"x": 63.46666666666667, "y": 53.86666666666666,
                                     "width": 28.933333333333334,
                                     "height": 17.733333333333334, "labels": ["Handwriting"],
                                     "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080}]}
    result2 = {"result": [{"id": "rWSy2ZUZm7", "type": "rectangle",
                           "value": {"x": 26.000000000000036, "y": 31.6, "width": 13.066666666666638,
                                     "height": 14.133333333333297, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "bbox", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "rWSy2ZUZm7", "type": "labels",
                           "value": {"x": 26.000000000000036, "y": 31.6,
                                     "width": 13.066666666666638,
                                     "height": 14.133333333333297, "labels": ["Text"],
                                     "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 1080,
                           "original_height": 1080},
                          {"id": "ePS5El8SPd", "type": "rectangle",
                           "value": {"x": 62.93333333333333, "y": 55.2, "width": 28.799999999999997, "height": 15.6,
                                     "rotation": 0}, "origin": "manual", "to_name": "image", "from_name": "bbox",
                           "image_rotation": 0, "original_width": 1080, "original_height": 1080}]}
    o1 = OCREvalItem(result1)
    o2 = OCREvalItem(result2)
    score = o1.compare(o2)
    assert score == 0.5


def test_ocr_2_groups_of_regions_without_rectangle_intersection():
    """
    DEV-1721
    Annotations with same labels and with Rectangles with no intersection
    :return:
    """
    result1 = {"result": [{"id": "7FDocDMHd1", "type": "rectangle",
                           "value": {"x": 10.3125, "y": 15, "width": 45.9375, "height": 23.333333333333332,
                                     "rotation": 0}, "origin": "manual", "to_name": "image", "from_name": "bbox",
                           "image_rotation": 0, "original_width": 320, "original_height": 240},
                          {"id": "7FDocDMHd1", "type": "labels",
                           "value": {"x": 10.3125, "y": 15, "width": 45.9375, "height": 23.333333333333332,
                                     "labels": ["Text"], "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 320, "original_height": 240},
                          {"id": "7FDocDMHd1", "type": "textarea",
                           "value": {"x": 10.3125, "y": 15, "text": ["Test Test"], "width": 45.9375,
                                     "height": 23.333333333333332, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "transcription", "image_rotation": 0, "original_width": 320,
                           "original_height": 240}]}
    result2 = {"result": [{"id": "7FDocDMHdj", "type": "rectangle",
                           "value": {"x": 18.4375, "y": 26.25, "width": 45.9375, "height": 23.333333333333332,
                                     "rotation": 0}, "origin": "manual", "to_name": "image", "from_name": "bbox",
                           "image_rotation": 0, "original_width": 320, "original_height": 240},
                          {"id": "7FDocDMHdj", "type": "labels",
                           "value": {"x": 18.4375, "y": 26.25, "width": 45.9375, "height": 23.333333333333332,
                                     "labels": ["Text"], "rotation": 0}, "origin": "manual", "to_name": "image",
                           "from_name": "label", "image_rotation": 0, "original_width": 320, "original_height": 240},
                          {"id": "7FDocDMHdj", "type": "textarea",
                           "value": {"x": 18.4375, "y": 26.25, "text": ["Test Test Test"], "width": 45.9375,
                                     "height": 23.333333333333332, "rotation": 0}, "origin": "manual",
                           "to_name": "image", "from_name": "transcription", "image_rotation": 0, "original_width": 320,
                           "original_height": 240}]}
    o1 = OCREvalItem(result1)
    o2 = OCREvalItem(result2)
    score = o1.compare(o2)
    assert score == 0
