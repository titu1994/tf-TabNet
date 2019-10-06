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

# Training Convenience
Due to model construction, Tensorflow autograph cannot yet trace the model properly. If you wish to avoid custom training loops, it is recommended to simply pass the `run_eagerly=True` flag to the `model.compile(...)` step.

You may also set `dynamic=True` when building the model, either via the base `TabNet` or either the Classification or Regression variants.

This step can be avoided if one is writing a custom training script.

# Mask Visualization
The masks of the TabNet can be obtained by using the TabNet class properties
 - `feature_selection_masks`: Returns a list of 1 or more masks at intermediate decision steps. Number of masks = number of decision steps - 1
 - `aggregate_feature_selection_mask`: Returns a single tensor which is the average activation of the masks over that batch of training samples.
 
 These masks can be obtained as `TabNet.feature_selection_masks`. Since the `TabNetClassification` and `TabNetRegression` models are composed of `TabNet`, the masks can be obtained as `model.tabnet.*`
 
These masks can be visualized in Tensorboard as follows - 
```python
writer = tf.summary.create_file_writer("logs/")
with writer.as_default():
    for i, mask in enumerate(model.tabnet.feature_selection_masks):
        print("Saving mask {} of shape {}".format(i + 1, mask.shape))
        tf.summary.image('mask_at_iter_{}'.format(i + 1), step=0, data=mask, max_outputs=1)
        writer.flush()

    agg_mask = model.tabnet.aggregate_feature_selection_mask
    print("Saving aggregate mask of shape", agg_mask.shape)
    tf.summary.image("Aggregate Mask", step=0, data=agg_mask, max_outputs=1)
    writer.flush()
writer.close()
```

# Requirements
- Tensorflow 2.0+ (1.14+ with V2 compat enabled may be sufficient for 1.x)
- Tensorflow-datasets (Only required for evaluating `iris.py`)
