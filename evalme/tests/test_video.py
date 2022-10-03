from evalme.video.video import VideoEvalItem

from text.text import simple_match_by_shape_key


def test_video_same_example():
    """
    Simple test with same data
    """
    example = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "framesCount": 10000000,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    },
                    {
                        "frame": 11,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 2.02
                    },
                    {
                        "frame": 15,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 3.10
                    }
                ]
            }
        }
    ]
    gt = VideoEvalItem(example)
    pred = VideoEvalItem(example)
    res = gt.iou_over_time(pred)
    res_label = gt.iou_over_time(pred, per_label=True)
    assert res == 1.0
    assert res_label == {'Test': 1.0}


def test_video_overlaping_frames_2_seq():
    """
    Overlapping video frames
    """
    example1 = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "framesCount": 10000000,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    },
                    {
                        "frame": 11,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 2.02
                    },
                    {
                        "frame": 15,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 3.10
                    }
                ]
            }
        }
    ]
    example2 = [
        {
            "id": "tBnMKLMC7G",
            "type": "videorectangle",
            "value": {
                "framesCount": 200,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    gt = VideoEvalItem(example1)
    pred = VideoEvalItem(example2)
    res = gt.iou_over_time(pred)
    res_label = gt.iou_over_time(pred, per_label=True)
    assert res == 0.5
    assert res_label == {'Test': 0.5}


def test_video_overlaping_frames_1_seq():
    """
    Overlapping video frames
    """
    example1 = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "framesCount": 4,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 4,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    example2 = [
        {
            "id": "tBnMKLMC7G",
            "type": "videorectangle",
            "value": {
                "framesCount": 4,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 2,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    gt = VideoEvalItem(example1)
    pred = VideoEvalItem(example2)
    res = gt.iou_over_time(pred)
    res_label = gt.iou_over_time(pred, per_label=True)
    assert res == 0.4167623421354765
    assert res_label == {'Test': 0.4167623421354765}


def test_video_overlaping_frames_1_seq_no_transform():
    """
    Overlapping video frames
    """
    example1 = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "framesCount": 4,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 4,
                        "enabled": False,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    example2 = [
        {
            "id": "tBnMKLMC7G",
            "type": "videorectangle",
            "value": {
                "framesCount": 4,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 2,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    gt = VideoEvalItem(example1)
    pred = VideoEvalItem(example2)
    res = gt.iou_over_time(pred)
    res_label = gt.iou_over_time(pred, per_label=True)
    assert res == 0.6
    assert res_label == {'Test': 0.6}


def test_video_not_overlaping_data():
    """
    Not overlapping video frames
    """
    example1 = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "framesCount": 10000000,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 11,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 2.02
                    },
                    {
                        "frame": 15,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 3.10
                    }
                ]
            }
        }
    ]
    example2 = [
        {
            "id": "tBnMKLMC7G",
            "type": "videorectangle",
            "value": {
                "framesCount": 200,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    gt = VideoEvalItem(example1)
    pred = VideoEvalItem(example2)
    res = gt.iou_over_time(pred)
    res_label = gt.iou_over_time(pred, per_label=True)
    assert res == 0
    assert res_label == {'Test': 0}


def test_video_different_labels():
    """
    Different labels in data
    """
    example1 = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "framesCount": 10000000,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 11,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 2.02
                    },
                    {
                        "frame": 15,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 3.10
                    }
                ]
            }
        }
    ]
    example2 = [
        {
            "id": "tBnMKLMC7G",
            "type": "videorectangle",
            "value": {
                "framesCount": 200,
                "labels": ['Test1'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    gt = VideoEvalItem(example1)
    pred = VideoEvalItem(example2)
    res = gt.iou_over_time(pred)
    res_label = gt.iou_over_time(pred, per_label=True)
    assert res == 0
    assert res_label == {'Test': 0}


def test_simple_video_metric():
    """
    Simple video metric test
    """
    example1 = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "framesCount": 10000000,
                "labels": ['Test'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    example2 = [
        {
            "id": "tBnMKLMC7G",
            "type": "videorectangle",
            "value": {
                "framesCount": 200,
                "labels": ['Test1'],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.01
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0,
                        "time": 1.55
                    }
                ]
            }
        }
    ]
    assert simple_match_by_shape_key(example1, example2, shape_key='sequence', check_condition=False) == 1.0
