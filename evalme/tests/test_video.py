import pytest

from evalme.video.object_detection import VideoObjectDetectionEvalItem

def test_simple_example():
    example = [
        {
            "id": "tJhYZLMC9G",
            "type": "videorectanglelabels",
            "value": {
                "videorectanglelabels": [
                    "Airplane"
                ],
                "sequence": [
                    {
                        "frame": 1,
                        "enabled": True,
                        "x": 38.266666666666666,
                        "y": 38.898756660746,
                        "width": 41.333333333333336,
                        "height": 22.202486678507995,
                        "rotation": 0
                    },
                    {
                        "frame": 5,
                        "enabled": False,
                        "x": 40.266666666666666,
                        "y": 49.898756660746,
                        "width": 41.333333333333336,
                        "height": 22.202486678507995,
                        "rotation": 0
                    }
                ]
            }
        }
    ]
    v = VideoObjectDetectionEvalItem(example)
    key_frames = v.extract_key_frames()
    assert len(key_frames) == 4
