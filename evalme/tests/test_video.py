import pytest

from evalme.video.object_detection import VideoObjectDetectionEvalItem

def test_video_disabled_till_end():
    """
    Test frames extraction with disabled in the end and frame count > disabled key frame
    """
    example = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "labels": [
                    "Airplane"
                ],
                "frameCount": 10000000,
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 56,
                        "height": 34,
                        "rotation": 30
                    }
                ],
                "from_name": "test"
            }
        }
    ]
    v = VideoObjectDetectionEvalItem(example)
    key_frames = v.extract_key_frames()
    assert len(key_frames) == 5
    assert key_frames[0]['value']['x'] == 38
    assert key_frames[0]['value']['y'] == 38
    assert key_frames[0]['value']['width'] == 41
    assert key_frames[0]['value']['height'] == 22
    assert key_frames[0]['value']['rotation'] == 0
    assert key_frames[0]['value']["from_name"] == "test"
    assert key_frames[4]['value']['x'] == 40
    assert key_frames[4]['value']['y'] == 49
    assert key_frames[4]['value']['width'] == 56
    assert key_frames[4]['value']['height'] == 34
    assert key_frames[4]['value']['rotation'] == 30
    assert key_frames[2]['value']['x'] == 39
    assert key_frames[2]['value']['y'] == 43.5
    assert key_frames[2]['value']['width'] == 48.5
    assert key_frames[2]['value']['height'] == 28
    assert key_frames[2]['value']['rotation'] == 15


def test_video_enabled_till_end():
    """
    Test frames extraction with enabled in the end and frame count > enabled key frame
    """
    example = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "labels": [
                    "Airplane"
                ],
                "frameCount": 10,
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    },
                    {
                        "frame": 5,
                        "enabled": True,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    }
                ]
            }
        }
    ]
    v = VideoObjectDetectionEvalItem(example)
    key_frames = v.extract_key_frames()
    assert len(key_frames) == 10
    assert key_frames[0]['value']['x'] == 38
    assert key_frames[0]['value']['y'] == 38
    assert key_frames[0]['value']['width'] == 41
    assert key_frames[0]['value']['height'] == 22
    assert key_frames[0]['value']['rotation'] == 0
    assert key_frames[4]['value']['x'] == 40
    assert key_frames[4]['value']['y'] == 49
    assert key_frames[4]['value']['width'] == 41
    assert key_frames[4]['value']['height'] == 22
    assert key_frames[4]['value']['rotation'] == 0
    assert key_frames[2]['value']['x'] == 39
    assert key_frames[2]['value']['y'] == 43.5
    assert key_frames[2]['value']['width'] == 41
    assert key_frames[2]['value']['height'] == 22
    assert key_frames[2]['value']['rotation'] == 0

    assert key_frames[5]['value']['x'] == 40
    assert key_frames[5]['value']['y'] == 49
    assert key_frames[5]['value']['width'] == 41
    assert key_frames[5]['value']['height'] == 22
    assert key_frames[5]['value']['rotation'] == 0
    assert key_frames[8]['value']['x'] == 40
    assert key_frames[8]['value']['y'] == 49
    assert key_frames[8]['value']['width'] == 41
    assert key_frames[8]['value']['height'] == 22
    assert key_frames[8]['value']['rotation'] == 0
    assert key_frames[9]['value']['x'] == 40
    assert key_frames[9]['value']['y'] == 49
    assert key_frames[9]['value']['width'] == 41
    assert key_frames[9]['value']['height'] == 22
    assert key_frames[9]['value']['rotation'] == 0


def test_video_enabled_till_end_one_frame():
    """
    Test frames extraction with enabled in the end and frame count > enabled key frame
    """
    example = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "labels": [
                    "Airplane",
                    "Test"
                ],
                "frameCount": 10,
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    }
                ]
            }
        }
    ]
    v = VideoObjectDetectionEvalItem(example)
    key_frames = v.extract_key_frames()
    print(key_frames)
    assert len(key_frames) == 10
    assert key_frames[0]['value']['x'] == 38
    assert key_frames[0]['value']['y'] == 38
    assert key_frames[0]['value']['width'] == 41
    assert key_frames[0]['value']['height'] == 22
    assert key_frames[0]['value']['rotation'] == 0
    assert key_frames[9]['value']['x'] == 38
    assert key_frames[9]['value']['y'] == 38
    assert key_frames[9]['value']['width'] == 41
    assert key_frames[9]['value']['height'] == 22
    assert key_frames[9]['value']['rotation'] == 0


def test_video_disabled_till_end_one_frame():
    """
    Test frames extraction with disabled in the end and frame count > enabled key frame
    """
    example = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "labels": [
                    "Airplane"
                ],
                "frameCount": 10,
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": False,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    }
                ]
            }
        }
    ]
    v = VideoObjectDetectionEvalItem(example)
    key_frames = v.extract_key_frames()
    print(key_frames)
    assert len(key_frames) == 0


def test_video_disabled_till_end():
    """
    Test frames extraction with disabled in the end and frame count > disabled key frame
    """
    example = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectangle",
            "value": {
                "labels": [
                    "Airplane"
                ],
                "frameCount": 10000000,
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    },
                    {
                        "frame": 11,
                        "enabled": True,
                        "x": 38,
                        "y": 38,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    },
                    {
                        "frame": 15,
                        "enabled": False,
                        "x": 40,
                        "y": 49,
                        "width": 41,
                        "height": 22,
                        "rotation": 0
                    }
                ]
            }
        }
    ]
    v = VideoObjectDetectionEvalItem(example)
    key_frames = v.extract_key_frames()
    assert len(key_frames) == 10
    assert key_frames[5]['value']['x'] == 38
    assert key_frames[5]['value']['y'] == 38
    assert key_frames[5]['value']['width'] == 41
    assert key_frames[5]['value']['height'] == 22
    assert key_frames[5]['value']['rotation'] == 0
    assert key_frames[9]['value']['x'] == 40
    assert key_frames[9]['value']['y'] == 49
    assert key_frames[9]['value']['width'] == 41
    assert key_frames[9]['value']['height'] == 22
    assert key_frames[9]['value']['rotation'] == 0
    assert key_frames[7]['value']['x'] == 39
    assert key_frames[7]['value']['y'] == 43.5
    assert key_frames[7]['value']['width'] == 41
    assert key_frames[7]['value']['height'] == 22
    assert key_frames[7]['value']['rotation'] == 0