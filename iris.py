import tensorflow as tf
import tensorflow_datasets as tfds
import tabnet

BATCH_SIZE = 50


def transform(ds):
    features = tf.unstack(ds['features'])
    labels = ds['label']

    x = dict(zip(col_names, features))
    y = tf.one_hot(labels, 3)
    return x, y


col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
ds_train = tfds.load(name="iris", split=tfds.Split.TRAIN)
ds_train = ds_train.shuffle(150)
ds_train = ds_train.map(transform)
ds_train = ds_train.batch(BATCH_SIZE)

feature_columns = []
for col_name in col_names:
    feature_columns.append(tf.feature_column.numeric_column(col_name))

model = tabnet.TabNetClassification(feature_columns, num_classes=3,
                                    feature_dim=4, output_dim=4,
                                    num_decision_steps=2, relaxation_factor=1.0,
                                    sparsity_coefficient=1e-5, batch_momentum=0.98,
                                    virtual_batch_size=None)

lr = tf.keras.optimizers.schedules.InverseTimeDecay(0.01, decay_steps=50, decay_rate=0.5, staircase=True)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

model.fit(ds_train, epochs=100)

model.summary()
