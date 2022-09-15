import pytest

from evalme.metrics import Metrics


def test_labels_choices_default_weights():
    result_1 = [
        {
          "id": "bMeSjkaisC",
          "type": "labels",
          "value": {
            "end": 25,
            "text": "oting",
            "start": 20,
            "labels": [
              "MISC"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "bMeSjkaisC",
          "type": "choices",
          "value": {
            "end": 25,
            "text": "oting",
            "start": 20,
            "choices": [
              "Adult content"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        }
      ]
    result_2 = [
        {
          "id": "EKWgT1LXXG",
          "type": "labels",
          "value": {
            "end": 25,
            "text": "Police shooting",
            "start": 16,
            "labels": [
              "MISC"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "EKWgT1LXXG",
          "type": "choices",
          "value": {
            "end": 25,
            "text": "Police shooting",
            "start": 16,
            "choices": [
              "Adult content"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        }
      ]
    r1 = Metrics.apply({}, result_1, result_2)
    r2 = Metrics.apply({}, result_2, result_1)
    assert r1 > 0.555
    assert r2 > 0.555

    feature_flags = {}
    feature_flags['ff_back_dev_2762_textarea_weights_30062022_short'] = True
    r1 = Metrics.apply({}, result_1, result_2, feature_flags=feature_flags)
    r2 = Metrics.apply({}, result_2, result_1, feature_flags=feature_flags)

    assert r1 == 1
    assert r2 == 1

@pytest.mark.parametrize(
    "start, end, label_weight, choice_weight, agreement",
    [
        [20, 25, None, None, 0.5],
        [20, 25, 0, 1, 1],
        [20, 25, 1, 0, 0],
        [20, 25, 0, 0, 0],
        [20, 25, 1, 1, 0.5],
    ],
)
def test_labels_choices_with_weights(start, end, label_weight, choice_weight, agreement):
    result_1 = [
        {
          "id": "bMeSjkaisC",
          "type": "labels",
          "value": {
            "end": end,
            "text": "oting",
            "start": start,
            "labels": [
              "MISC"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "bMeSjkaisC",
          "type": "choices",
          "value": {
            "end": end,
            "text": "oting",
            "start": start,
            "choices": [
              "Adult content"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        }
      ]
    result_2 = [
        {
          "id": "EKWgT1LXXG",
          "type": "labels",
          "value": {
            "end": end,
            "text": "Police shooting",
            "start": start,
            "labels": [
              "ORG"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "EKWgT1LXXG",
          "type": "choices",
          "value": {
            "end": end,
            "text": "Police shooting",
            "start": start,
            "choices": [
              "Adult content"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        }
      ]

    feature_flags = {}
    feature_flags['ff_back_dev_2762_textarea_weights_30062022_short'] = True

    if label_weight is not None and choice_weight is not None:
        project_params = {'control_weights': {
            "label": {"type": "Labels", "labels": {"LOC": 1.0, "ORG": 1.0, "PER": 1.0, "MISC": 1.0}, "overall": label_weight},
            "choice": {"type": "Choices", "labels": {"Weapons": 1.0, "Violence": 1.0, "Adult content": 1.0},
                       "overall": choice_weight}}}
    else:
        project_params = {}

    r1 = Metrics.apply(project_params, result_1, result_2, feature_flags=feature_flags)
    r2 = Metrics.apply(project_params, result_2, result_1, feature_flags=feature_flags)

    assert r1 == agreement
    assert r2 == agreement


@pytest.mark.parametrize(
    "start, end, label_weight, choice_weight, agreement",
    [
        [20, 25, None, None, 0],
        [20, 25, 0, 1, 0],
        [20, 25, 1, 0, 0],
        [20, 25, 0, 0, 0],
        [20, 25, 1, 1, 0],
    ],
)
def test_labels_choices_with_weights_not_matching_regions(start, end, label_weight, choice_weight, agreement):
    result_1 = [
        {
          "id": "bMeSjkaisC",
          "type": "labels",
          "value": {
            "end": end,
            "text": "oting",
            "start": start,
            "labels": [
              "MISC"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "bMeSjkaisC",
          "type": "choices",
          "value": {
            "end": end,
            "text": "oting",
            "start": start,
            "choices": [
              "Adult content"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        }
      ]
    result_2 = [
        {
          "id": "EKWgT1LXXG",
          "type": "labels",
          "value": {
            "end": end+end,
            "text": "Police shooting",
            "start": start+start,
            "labels": [
              "ORG"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "label"
        },
        {
          "id": "EKWgT1LXXG",
          "type": "choices",
          "value": {
            "end": end+end,
            "text": "Police shooting",
            "start": start+start,
            "choices": [
              "Adult content"
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "choice"
        }
      ]

    feature_flags = {}
    feature_flags['ff_back_dev_2762_textarea_weights_30062022_short'] = True

    if label_weight is not None and choice_weight is not None:
        project_params = {'control_weights': {
            "label": {"type": "Labels", "labels": {"LOC": 1.0, "ORG": 1.0, "PER": 1.0, "MISC": 1.0}, "overall": label_weight},
            "choice": {"type": "Choices", "labels": {"Weapons": 1.0, "Violence": 1.0, "Adult content": 1.0},
                       "overall": choice_weight}}}
    else:
        project_params = {}

    r1 = Metrics.apply(project_params, result_1, result_2, feature_flags=feature_flags)
    r2 = Metrics.apply(project_params, result_2, result_1, feature_flags=feature_flags)

    assert r1 == agreement
    assert r2 == agreement


@pytest.mark.parametrize(
    "x, y, w, h, label_weight, text_weight, agreement",
    [
        [20, 25, 20, 25, None, None, 0.5],
        [20, 25, 20, 25, 0, 1, 1],
        [20, 25, 20, 25, 1, 0, 0],
        [20, 25, 20, 25, 0, 0, 0],
        [20, 25, 20, 25, 1, 1, 0.5],
        [20, 25, 20, 25, 0.15, 1, 0.8695652173913044]
    ],
)
def test_labels_textarea(x, y, w, h, label_weight, text_weight, agreement):
    result_1 = [
        {
          "id": "sP9pbTYATu",
          "type": "rectanglelabels",
          "value": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rotation": 0,
            "rectanglelabels": [
              "Airplane"
            ]
          },
          "origin": "manual",
          "to_name": "image",
          "from_name": "label",
          "image_rotation": 0,
          "original_width": 768,
          "original_height": 578
        },
        {
          "id": "sP9pbTYATu",
          "type": "textarea",
          "value": {
            "x": x,
            "y": y,
            "text": [
              "11/11/2022"
            ],
            "width": w,
            "height": h,
            "rotation": 0
          },
          "origin": "manual",
          "to_name": "image",
          "from_name": "date_bm",
          "image_rotation": 0,
          "original_width": 768,
          "original_height": 578
        }
      ]
    result_2 = [
        {
          "id": "sP9pbTYATu",
          "type": "rectanglelabels",
          "value": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rotation": 0,
            "rectanglelabels": [
              "Car"
            ]
          },
          "origin": "manual",
          "to_name": "image",
          "from_name": "label",
          "image_rotation": 0,
          "original_width": 768,
          "original_height": 578
        },
        {
          "id": "sP9pbTYATu",
          "type": "textarea",
          "value": {
            "x": x,
            "y": y,
            "text": [
              "11/11/2022"
            ],
            "width": w,
            "height": h,
            "rotation": 0
          },
          "origin": "manual",
          "to_name": "image",
          "from_name": "date_bm",
          "image_rotation": 0,
          "original_width": 768,
          "original_height": 578
        }
      ]
    feature_flags = {}
    feature_flags['ff_back_dev_2762_textarea_weights_30062022_short'] = True

    if label_weight is not None and text_weight is not None:
        project_params = {'control_weights': {"label": {"type": "RectangleLabels", "labels": {"Car": 1.0, "Airplane": 1.0}, "overall": label_weight},
                                              "date_bm": {"type": "TextArea", "labels": {}, "overall": text_weight}}}
    else:
        project_params = {}

    r1 = Metrics.apply(project_params, result_1, result_2, feature_flags=feature_flags)
    r2 = Metrics.apply(project_params, result_2, result_1, feature_flags=feature_flags)

    assert r1 == agreement
    assert r2 == agreement


def test_labels_number():
    pass


def test_labels_datetime():
    pass


def test_labels_choices_textarea_number_datetime():
    pass