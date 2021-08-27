import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import tabnet

tf.random.set_seed(0)
train_size = 1500
BATCH_SIZE = 500
num_classes = 50


def transform(_):
    # Randomly sample an integer in the range of the vocabulary range [0, 1000]
    features = tf.random.uniform([], 0, 1001, dtype=tf.int32)

    # Its class will be the sampled value % the number of classes
    labels = tf.math.mod(features, num_classes)

    # Then map to a dummy column called "data_col" and one hot encode the label.
    x = {'data_col': features}
    y = tf.one_hot(labels, num_classes)
    return x, y


ds_full = tf.data.Dataset.range(2000)  # 2000 samples
ds_full = ds_full.shuffle(2000, seed=0)

# Note: Train and test are drawn from same distribution for demonstration purposes.
# We should get near identical scores on both of them.
ds_train = ds_full.take(train_size)
ds_train = ds_train.map(transform,num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_full.skip(train_size)
ds_test = ds_test.map(transform,num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)


class EmbeddingTabNet(tf.keras.Model):
    """
    Here we map a categorical value with relatively high cardinality to an embedding of dimension 10.
    The above embedding cost is a matrix of shape (number of tokens in vocabulary (num of ids), embedding dimension).
    Here it should cost (1001, 10) = 10010 parameters.

    Then we project this embedding to a 512 length vector.
    This should be done by a matrix of shape (10, 512) = 5120 parameters.

    The final "classifier" is a combination of TabNet and the mapping of its final state
    the number of classes. Its parameter count should be approximately 5500 through rough calculation.
    """
    def __init__(self, embedding_dim, projection_dim, vocab_size=1000, **kwargs):
        super(EmbeddingTabNet, self).__init__(**kwargs)

        # For demonstration purposes, assume we have a "data_col" column.
        cat_col = tf.feature_column.categorical_column_with_identity("data_col", num_buckets=vocab_size + 1,
                                                                     default_value=0)
        embed_col = tf.feature_column.embedding_column(cat_col, embedding_dim)
        self.embed_layer = tf.keras.layers.DenseFeatures([embed_col])
        self.projection = tf.keras.layers.Dense(projection_dim, activation='linear', use_bias=False)

        # Assume we have `d` classes, mapped as y = (data_col % d), where data col can take 1000 values.
        # Note: `num_features` *must* be the length of the projection dim.
        self.tabnet_model = tabnet.TabNetClassifier(None, num_classes=num_classes, num_features=projection_dim,
                                                    feature_dim=4, output_dim=4,
                                                    num_decision_steps=2, relaxation_factor=1.0,
                                                    sparsity_coefficient=1e-5, batch_momentum=0.98,
                                                    virtual_batch_size=None, norm_type='group',
                                                    num_groups=2)

    def call(self, inputs, training=None):
        embed = self.embed_layer(inputs)  # Map integer index to an embedding vector; Shape = (None, embedding_size)
        proj = self.projection(embed)  # Project embedding to higher dimensional space; Shape = (None, projection_size)
        out = self.tabnet_model(proj, training=training)
        return out

    @property
    def tabnet(self):
        # Used for visualization to directly access the tabnet model underlying the model.
        return self.tabnet_model.tabnet


model = EmbeddingTabNet(embedding_dim=10, projection_dim=512, vocab_size=1000)


lr = tf.keras.optimizers.schedules.ExponentialDecay(0.05, decay_steps=100, decay_rate=0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(ds_train, epochs=200, validation_data=ds_test, verbose=2)

model.summary()

print()
if os.path.exists('embedding_logs/'):
    shutil.rmtree('embedding_logs/')

""" Save the images of the feature masks """
# Force eager execution mode to generate the masks
x, y = next(iter(ds_train))
_ = model(x)

writer = tf.summary.create_file_writer("embedding_logs/")
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
