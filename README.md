# TabNet for Tensorflow 2.0
A Tensorflow 2.0 port for the paper [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442), whose original codebase is available at https://github.com/google-research/google-research/blob/master/tabnet.

<img src="https://github.com/titu1994/tf-TabNet/blob/master/images/TabNet.png?raw=true" height=100% width=100%>

The above image is obtained from the paper, where the model is built of blocks in two stages - one to attend to the input features and anither to construct the output of the model. 

# Usage

The script `tabnet.py` can be imported to yield either the `TabNet` building block, or the `TabNetClassification` and `TabNetRegression` models, which add appropriate heads for the basic `TabNet` model. If the classification or regression head is to be customized, it is recommended to compose a new model with the `TabNet` as the base of the model.

```python
from tabnet import TabNet, TabNetClassification

model = TabNetClassification(feature_list, num_classes, ...)
```

As the models use custom objects, it is necessary to import `custom_objects.py` in an evaluation only script.

# Model Convenience
Due to model construction, Tensorflow autograph cannot yet trace the model properly. If you wish to avoid custom training loops, it is recommended to simply pass the `run_eagerly=True` flag to the `model.compile(...)` step.

You may also set `dynamic=True` when building the model, either via the base `TabNet` or either the Classification or Regression variants.

This step can be avoided if one is writing a custom training script.

# Requirements
- Tensorflow 2.0+ (1.14+ with V2 compat enabled may be sufficient for 1.x)
- Tensorflow-datasets (Only required for evaluating `iris.py`)
