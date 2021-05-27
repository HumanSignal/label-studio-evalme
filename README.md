# label-studio-evalme
Evaluation metrics package

## Installation

Simple installation from PyPI
```bash
pip install label-studio-evalme
```

<details>
  <summary>Other installations</summary>
	Pip from source
	```bash
	# with git
	pip install git+https://github.com/heartexlabs/label-studio-evalme.git@master
	```

</details>

## What is Evalme
Evalme is a collection of Label Studio evaluation metrics implementations and an easy-to-use API to create custom metrics. It offers:

* A standardized interface to increase reproducibility
* Reduces boilerplate
* Metrics optimized for Label Studio format

You can use Evalme with any Label Studio verions or with Label Studio Enterprise.


## Using Evalme

### Loading existing data from Label Studio

You can use Label Studio REST API to load existing data from your instance of Label Studio or Label Studio Enterprise.

Specify your Label Studio url, access token and project id in the parametres:
``` python
from evalme.matcher import Matcher

loader = Matcher(url="http://127.0.0.1:8000",
                 token="ACCESS_TOKEN",
                 project='1')
loader.refresh()
```

You can load data from file exported from [Label Studio](https://labelstud.io/guide/api.html#Export-annotations):
``` python
from evalme.matcher import Matcher

loader = Matcher()
loader.load('your_filename')
```

Data is available in _raw_data field. 

### Built-in metrics

By default there is a naive metric object. It evaluates difference with naive approach:
if object is fully equals to the other one evaluation method returns 1,
else it returns 0.

Using built-in metric case:

``` python
from evalme.matcher import Matcher

loader = Matcher()
loader.load('your_filename')
# Run agreement_matrix method to get matrix for all your annotations
matrix = loader.agreement_matrix()
# print result
print(matrix)
```

### Implementing your own metric

Implementing your own metric is easy - create an evalution function and register it in Metrics class. 
Simply, create an evaluation function with 2 parametres for compared objects.

```python
from evalme.matcher import Matcher
# write your own evaluation function or use existing one
def naive(x, y):
	"""
    Naive comparison of annotations
    """
    if len(x) != len(y):
        result = 0
    else:
        for i in range(len(x)):
            if x[i]['value'] != y[i]['value']:
                result = 0
                break
        else:
            result = 1
    return result
# Register it in Metrics object
Metrics.register(
    name='naive',
    form=None,
    tag='all',
    func=naive,
    desc='Naive comparison of result dict'
)
# create Matcher object from previous example
loader = Matcher()
loader.load('your_filename')
matrix = loader.agreement_matrix(metric_name='naive')
# print result
print(matrix)
```

## Contribute!
The Label Studio team is hard at work adding even more metrics.
But we're looking for incredible contributors like you to submit new metrics
and improve existing ones!

Join our [Slack](https://join.slack.com/t/label-studio/shared_invite/zt-cr8b7ygm-6L45z7biEBw4HXa5A2b5pw)
to get help becoming a contributor!

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/label-studio/shared_invite/zt-cr8b7ygm-6L45z7biEBw4HXa5A2b5pw)!

## License
Please observe the MIT License that is listed in this repository. 
