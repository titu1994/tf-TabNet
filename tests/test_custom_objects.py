import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
import tabnet.custom_objects as custom_objs


def deterministic(func):
    def wrapper(*args, **kwargs):
        # Set seeds
        np.random.seed(0)

        if hasattr(tf, 'set_random_seed'):
            tf.set_random_seed(0)
        else:
            tf.random.set_seed(0)

        output = func(*args, **kwargs)
        return output
    return wrapper


def _test_random_shape_on_all_axis_except_batch(shape, groups,
                                                center, scale):
    inputs = tf.random.normal((shape))
    for axis in range(1, len(shape)):
        _test_specific_layer(inputs, axis, groups, center, scale)


def _test_specific_layer(inputs, axis, groups, center, scale):

    input_shape = inputs.shape

    # Get Output from Keras model
    layer = custom_objs.GroupNormalization(
        axis=axis, groups=groups, center=center, scale=scale)
    model = tf.keras.models.Sequential()
    model.add(layer)
    outputs = model.predict(inputs)
    assert not (np.isnan(outputs).any())

    # Create shapes
    if groups is -1:
        groups = input_shape[axis]
    np_inputs = inputs.numpy()
    reshaped_dims = list(np_inputs.shape)
    reshaped_dims[axis] = reshaped_dims[axis] // groups
    reshaped_dims.insert(1, groups)
    reshaped_inputs = np.reshape(np_inputs, tuple(reshaped_dims))

    # Calculate mean and variance
    mean = np.mean(
        reshaped_inputs,
        axis=tuple(range(2, len(reshaped_dims))),
        keepdims=True)
    variance = np.var(
        reshaped_inputs,
        axis=tuple(range(2, len(reshaped_dims))),
        keepdims=True)

    # Get gamma and beta initalized by layer
    gamma, beta = layer._get_reshaped_weights(input_shape)
    if gamma is None:
        gamma = 1.0
    if beta is None:
        beta = 0.0

    # Get ouput from Numpy
    zeroed = reshaped_inputs - mean
    rsqrt = 1 / np.sqrt(variance + 1e-5)
    output_test = gamma * zeroed * rsqrt + beta

    # compare outputs
    output_test = np.reshape(output_test, input_shape.as_list())
    assert np.allclose(np.mean(output_test - outputs), 0., atol=1e-7, rtol=1e-7)


def _create_and_fit_Sequential_model(layer, shape):
    # Helperfunction for quick evaluation
    model = tf.keras.models.Sequential()
    model.add(layer)
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(0.01),
        loss="categorical_crossentropy")
    layer_shape = (10,) + shape
    input_batch = np.random.rand(*layer_shape)
    output_batch = np.random.rand(*(10, 1))
    model.fit(x=input_batch, y=output_batch, epochs=1, batch_size=1)
    return model


@deterministic
@run_all_in_graph_and_eager_modes
def test_glu():
    # Test with vector input
    x = tf.random.uniform([10], -1., 1.)
    out = custom_objs.glu(x, n_units=None)

    assert tf.reduce_mean(out) < 0.

    # Test with matrix input
    x = tf.random.uniform([10, 10], -1., 1.)
    out = custom_objs.glu(x, n_units=None)

    assert tf.reduce_mean(out) < 0.

    # Test with matrix input and n_units
    with pytest.raises(tf.errors.InvalidArgumentError):
        out2 = custom_objs.glu(x, n_units=2)  # needs to have equal partition of vectors


@deterministic
@run_all_in_graph_and_eager_modes
def test_sparsemax():
    x = tf.random.uniform([5, 20], -1., 1.)
    out = custom_objs.sparsemax(x, axis=-1)

    row_zero_sum = tf.reduce_sum(tf.cast(tf.equal(out, 0.0), tf.float32), axis=-1)
    assert tf.reduce_all(row_zero_sum > 0.)


@run_all_in_graph_and_eager_modes
def test_reshape():
    def run_reshape_test(axis, group, input_shape, expected_shape):
        group_layer = custom_objs.GroupNormalization(groups=group, axis=axis)
        group_layer._set_number_of_groups_for_instance_norm(input_shape)

        inputs = np.ones(input_shape)
        tensor_input_shape = tf.convert_to_tensor(input_shape)
        reshaped_inputs, group_shape = group_layer._reshape_into_groups(
            inputs, (10, 10, 10), tensor_input_shape)
        for i in range(len(expected_shape)):
            assert (int(group_shape[i]) == expected_shape[i])

    input_shape = (10, 10, 10)
    expected_shape = [10, 5, 10, 2]
    run_reshape_test(2, 5, input_shape, expected_shape)

    input_shape = (10, 10, 10)
    expected_shape = [10, 2, 5, 10]
    run_reshape_test(1, 2, input_shape, expected_shape)

    input_shape = (10, 10, 10)
    expected_shape = [10, 10, 1, 10]
    run_reshape_test(1, -1, input_shape, expected_shape)

    input_shape = (10, 10, 10)
    expected_shape = [10, 1, 10, 10]
    run_reshape_test(1, 1, input_shape, expected_shape)


@deterministic
@run_all_in_graph_and_eager_modes
def test_feature_input():
    shape = (10, 100)
    for center in [True, False]:
        for scale in [True, False]:
            for groups in [-1, 1, 2, 5]:
                _test_random_shape_on_all_axis_except_batch(
                    shape, groups, center, scale)


@deterministic
@run_all_in_graph_and_eager_modes
def test_picture_input():
    shape = (10, 30, 30, 3)
    for center in [True, False]:
        for scale in [True, False]:
            for groups in [-1, 1, 3]:
                _test_random_shape_on_all_axis_except_batch(
                    shape, groups, center, scale)


@deterministic
@run_all_in_graph_and_eager_modes
def test_weights():
    # Check if weights get initialized correctly
    layer = custom_objs.GroupNormalization(groups=1, scale=False, center=False)
    layer.build((None, 3, 4))
    assert (len(layer.trainable_weights) == 0)
    assert (len(layer.weights) == 0)


@deterministic
@run_all_in_graph_and_eager_modes
def test_apply_normalization():

    input_shape = (1, 4)
    expected_shape = (1, 2, 2)
    reshaped_inputs = tf.constant([[[2.0, 2.0], [3.0, 3.0]]])
    layer = custom_objs.GroupNormalization(groups=2, axis=1, scale=False, center=False)
    normalized_input = layer._apply_normalization(reshaped_inputs,
                                                  input_shape)
    assert (
        tf.reduce_all(
            tf.equal(normalized_input,
                     tf.constant([[[0.0, 0.0], [0.0, 0.0]]]))))


@deterministic
@run_all_in_graph_and_eager_modes
def test_axis_error():

    with pytest.raises(ValueError):
        custom_objs.GroupNormalization(axis=0)


@deterministic
@run_all_in_graph_and_eager_modes
def test_groupnorm_flat():
    # Check basic usage of groupnorm_flat
    # Testing for 1 == LayerNorm, 16 == GroupNorm, -1 == InstanceNorm

    groups = [-1, 16, 1]
    shape = (64,)
    for i in groups:
        model = _create_and_fit_Sequential_model(
            custom_objs.GroupNormalization(groups=i), shape)
        assert (hasattr(model.layers[0], 'gamma'))
        assert (hasattr(model.layers[0], 'beta'))


@deterministic
@run_all_in_graph_and_eager_modes
def test_initializer():
    # Check if the initializer for gamma and beta is working correctly

    layer = custom_objs.GroupNormalization(
        groups=32,
        beta_initializer='random_normal',
        beta_constraint='NonNeg',
        gamma_initializer='random_normal',
        gamma_constraint='NonNeg')

    model = _create_and_fit_Sequential_model(layer, (64,))

    weights = np.array(model.layers[0].get_weights())
    negativ = weights[weights < 0.0]
    assert (len(negativ) == 0)


@deterministic
@run_all_in_graph_and_eager_modes
def test_regularizations():

    layer = custom_objs.GroupNormalization(
        gamma_regularizer='l1', beta_regularizer='l1', groups=4, axis=2)
    layer.build((None, 4, 4))
    assert (len(layer.losses) == 2)
    max_norm = tf.keras.constraints.max_norm
    layer = custom_objs.GroupNormalization(
        gamma_constraint=max_norm, beta_constraint=max_norm)
    layer.build((None, 3, 4))
    assert (layer.gamma.constraint == max_norm)
    assert (layer.beta.constraint == max_norm)


@deterministic
@run_all_in_graph_and_eager_modes
def test_groupnorm_conv():
    # Check if Axis is working for CONV nets
    # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm

    groups = [-1, 5, 1]
    for i in groups:
        model = tf.keras.models.Sequential()
        model.add(
            custom_objs.GroupNormalization(axis=1, groups=i, input_shape=(20, 20, 3)))
        model.add(tf.keras.layers.Conv2D(5, (1, 1), padding='same'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.01), loss='mse')
        x = np.random.randint(1000, size=(10, 20, 20, 3))
        y = np.random.randint(1000, size=(10, 1))
        a = model.fit(x=x, y=y, epochs=1)
        assert (hasattr(model.layers[0], 'gamma'))


if __name__ == '__main__':
    pytest.main(__file__)
