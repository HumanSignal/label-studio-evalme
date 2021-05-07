from evalme.text.text import intersection_html_tagging


def test_paragraphlabels_1():
    a1 = [
        {
            "to_name": "text",
            "from_name": "hotel_star_rating_values",
            "type": "choices",
            "id": "Zofkvvz6e9",
            "value": {
                "start": "1",
                "endOffset": 65,
                "end": "1",
                "startOffset": 37,
                "choices": [
                    "4"
                ]
            }
        },
        {
            "to_name": "text",
            "from_name": "hotel_star_rating_values",
            "type": "choices",
            "id": "LDFbkdjLSi",
            "value": {
                "start": "5",
                "endOffset": 81,
                "end": "5",
                "startOffset": 57,
                "choices": [
                    "1"
                ]
            }
        }
    ]
    a2 = [
    {
        "to_name": "text",
        "from_name": "hotel_star_rating_values",
        "type": "choices",
        "id": "HvGBoa1nv4",
        "value": {
            "start": "1",
            "endOffset": 65,
            "end": "1",
            "startOffset": 37,
            "choices": [
                "4"
            ]
        }
    },
        {
            "to_name": "text",
            "from_name": "hotel_star_rating_values",
            "type": "choices",
            "id": "LDFbkdjLSi",
            "value": {
                "start": "5",
                "endOffset": 81,
                "end": "5",
                "startOffset": 57,
                "choices": [
                    "2"
                ]
            }
        }
    ]

    # half match because 2nd choice is "2" not "1"
    assert intersection_html_tagging(a1, a2, shape_key='choices') == 0.5
