# label-studio-evalme
Evaluation metrics package

## Installation

Simple installation from PyPI
```bash
pip install label-studio-evalme
```

<details>
  <summary>Other installation methods</summary>
  
	Pip from source
	```bash
	# with git
	pip install git+https://github.com/heartexlabs/label-studio-evalme.git@master
	```

</details>

## What is Evalme?
Evalme is a collection of Label Studio evaluation metric implementations and an easy-to-use API to create custom metrics. It offers:

* A standardized interface to increase reproducibility
* Reduced boilerplate
* Optimized metrics for Label Studio

## Get started with Evalme
You can use Evalme with any Label Studio versions or with Label Studio Enterprise.

### Load existing data from Label Studio

Use the Label Studio REST API to load existing data from your instance of Label Studio or Label Studio Enterprise.

Specify your Label Studio URL, access token and project ID in the parameters:
``` python
from evalme.matcher import Matcher

loader = Matcher(url="http://127.0.0.1:8000",
                 token="ACCESS_TOKEN",
                 project='1')
loader.refresh()
```

You can also load data from exported annotation files from Label Studio, exported using [the API](https://labelstud.io/guide/api.html#Export-annotations) or the [Label Studio UI](https://labelstud.io/guide/export.html):
``` python
from evalme.matcher import Matcher

loader = Matcher()
loader.load('your_filename')
```

After you load data, it is available in the `_raw_data` field. 

### Built-in metrics

By default there is a naive metric object. It evaluates annotation differences with a naive approach:
if an object is fully equal to another one, the evaluation method returns 1,
otherwise it returns 0.

To use the built-in metrics, do the following:

``` python
from evalme.matcher import Matcher

loader = Matcher()
loader.load('your_filename')
# Run agreement_matrix method to get matrix for all your annotations
matrix = loader.agreement_matrix()
# print result
print(matrix)
```

### Implement your own metric

You can implement your own metric by creating an evaluation function and registering it in Metrics class. 

For example, create an evaluation function with 2 parameters for compared objects:

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
The Label Studio team is hard at work adding even more metrics, but we're looking for incredible contributors like you to submit new metrics and improve existing ones!

Join our [Slack community](https://join.slack.com/t/label-studio/shared_invite/zt-cr8b7ygm-6L45z7biEBw4HXa5A2b5pw)
to get help becoming a contributor!

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/label-studio/shared_invite/zt-cr8b7ygm-6L45z7biEBw4HXa5A2b5pw)!

## License
Please observe the MIT License that is listed in this repository. 
