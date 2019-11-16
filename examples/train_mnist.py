import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import tabnet

if not os.path.exists('mnist'):
    os.makedirs('mnist')

BATCH_SIZE = 128


def transform(ds):
    features = ds['image']
    labels = ds['label']

    x = tf.reshape(features, [-1])
    x = tf.cast(x, tf.float32) / 255.
    y = tf.one_hot(labels, 10)
    return x, y


ds_train, ds_test = tfds.load(name="mnist", split=[tfds.Split.TRAIN, tfds.Split.TEST], data_dir='mnist',
                              shuffle_files=False)
ds_train = ds_train.shuffle(60000)
ds_train = ds_train.map(transform)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_test.map(transform)
ds_test = ds_test.batch(BATCH_SIZE)

# Use Group Normalization for small batch sizes
model = tabnet.TabNetClassifier(feature_columns=None, num_classes=10, num_features=784,
                                feature_dim=16, output_dim=16,
                                num_decision_steps=5, relaxation_factor=1.5,
                                sparsity_coefficient=0., batch_momentum=0.98,
                                virtual_batch_size=None, norm_type='group',
                                num_groups=-1)

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=500, decay_rate=0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(ds_train, epochs=5, validation_data=ds_test, verbose=2)

model.summary()

print()
if os.path.exists('mnist_logs/'):
    shutil.rmtree('mnist_logs/')

""" Save the images of the feature masks """
# Force eager execution mode to generate the masks
x, y = next(iter(ds_train))
_ = model(x)

writer = tf.summary.create_file_writer("mnist_logs/")
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
