import pytest

from evalme.classification import ClassificationEvalItem, ChoicesEvalItem, naive, exact_matching_choices

from evalme.metrics import Metrics


def test_dynamic_choices_not_matching():
    """
    Dynamic choices: not matching choices
    """
    result1 = {"result": [
        {
            "id": "zMpKg1F1Go",
            "type": "choices",
            "value": {
                "choices": [
                    [
                        "Products",
                        "Loan Payment Center"
                    ]
                ]
            },
            "origin": "manual",
            "to_name": "text",
            "from_name": "dynamic_choices"
        }
    ]}
    result2 = {"result": [
        {
          "id": "W50VSmRHl5",
          "type": "choices",
          "value": {
            "choices": [
              [
                "Products",
                "Personal Loans"
              ]
            ]
          },
          "origin": "manual",
          "to_name": "text",
          "from_name": "dynamic_choices"
        }
      ]
    }

    obj1 = ChoicesEvalItem(raw_data=result1['result'])
    obj2 = ChoicesEvalItem(raw_data=result2['result'])

    assert obj1.exact_match(obj2) == 0
    per_label = obj1.exact_match(obj2, per_label=True)
    assert per_label == {"Products": 0, "Loan Payment Center": 0}

    assert obj2.exact_match(obj1) == 0
    per_label = obj2.exact_match(obj1, per_label=True)
    assert per_label == {"Products": 0, "Personal Loans": 0}


def test_dynamic_choices_matching():
    """
    Dynamic choices: matching choices
    """
    result1 = {"result": [
        {
            "id": "zMpKg1F1Go",
            "type": "choices",
            "value": {
                "choices": [
                    [
                        "Products",
                        "Loan Payment Center"
                    ]
                ]
            },
            "origin": "manual",
            "to_name": "text",
            "from_name": "dynamic_choices"
        }
    ]}
    result2 = {"result": [
        {
            "id": "W50VSmRHl5",
            "type": "choices",
            "value": {
                "choices": [
                    [
                        "Products",
                        "Loan Payment Center"
                    ]
                ]
            },
            "origin": "manual",
            "to_name": "text",
            "from_name": "dynamic_choices"
        }
    ]
    }

    obj1 = ChoicesEvalItem(raw_data=result1['result'])
    obj2 = ChoicesEvalItem(raw_data=result2['result'])

    assert obj1.exact_match(obj2) == 1
    per_label = obj1.exact_match(obj2, per_label=True)
    assert per_label == {"Products": 1, "Loan Payment Center": 1}

    assert obj2.exact_match(obj1) == 1
    per_label = obj2.exact_match(obj1, per_label=True)
    assert per_label == {"Products": 1, "Loan Payment Center": 1}