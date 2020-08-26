import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import tabnet

train_size = 125
BATCH_SIZE = 50


def transform(ds):
    features = tf.unstack(ds['features'])
    labels = ds['label']

    x = dict(zip(col_names, features))
    y = tf.one_hot(labels, 3)
    return x, y


col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
ds_full = tfds.load(name="iris", split=tfds.Split.TRAIN)
ds_full = ds_full.shuffle(150, seed=0)

ds_train = ds_full.take(train_size)
ds_train = ds_train.map(transform)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_full.skip(train_size)
ds_test = ds_test.map(transform)
ds_test = ds_test.batch(BATCH_SIZE)

feature_columns = []
for col_name in col_names:
    feature_columns.append(tf.feature_column.numeric_column(col_name))

# Group Norm does better for small datasets
model = tabnet.TabNetClassifier(feature_columns, num_classes=3,
                                feature_dim=8, output_dim=4,
                                num_decision_steps=4, relaxation_factor=1.0,
                                sparsity_coefficient=1e-5, batch_momentum=0.98,
                                virtual_batch_size=None, norm_type='group',
                                num_groups=1)

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=100, decay_rate=0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(ds_train, epochs=100, validation_data=ds_test, verbose=2)

model.summary()

print()
if os.path.exists('logs/'):
    shutil.rmtree('logs/')

""" Save the images of the feature masks """
# Force eager execution mode to generate the masks
x, y = next(iter(ds_train))
_ = model(x)

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
