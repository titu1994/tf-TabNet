import tensorflow as tf
from tabnet.custom_objects import glu, sparsemax, GroupNormalization


class TransformBlock(tf.keras.Model):

    def __init__(self, features,
                 norm_type,
                 momentum=0.9,
                 virtual_batch_size=None,
                 groups=2,
                 block_name='',
                 **kwargs):
        super(TransformBlock, self).__init__(**kwargs)

        self.features = features
        self.norm_type = norm_type
        self.momentum = momentum
        self.groups = groups
        self.virtual_batch_size = virtual_batch_size

        self.transform = tf.keras.layers.Dense(self.features, use_bias=False, name=f'transformblock_dense_{block_name}')

        if norm_type == 'batch':
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                                         virtual_batch_size=virtual_batch_size,
                                                         name=f'transformblock_bn_{block_name}')

        else:
            self.bn = GroupNormalization(axis=-1, groups=self.groups, name=f'transformblock_gn_{block_name}')

    def call(self, inputs, training=None):
        x = self.transform(inputs)
        x = self.bn(x, training=training)
        return x


class TabNet(tf.keras.Model):

    def __init__(self, feature_columns,
                 feature_dim=64,
                 output_dim=64,
                 num_features=None,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=2,
                 epsilon=1e-5,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'batch' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNet, self).__init__(**kwargs)

        # Input checks
        if feature_columns is not None:
            if type(feature_columns) not in (list, tuple):
                raise ValueError("`feature_columns` must be a list or a tuple.")

            if len(feature_columns) == 0:
                raise ValueError("`feature_columns` must be contain at least 1 tf.feature_column !")

            if num_features is None:
                num_features = len(feature_columns)
            else:
                num_features = int(num_features)

        else:
            if num_features is None:
                raise ValueError("If `feature_columns` is None, then `num_features` cannot be None.")

        if num_decision_steps < 1:
            raise ValueError("Num decision steps must be greater than 0.")

        if feature_dim <= output_dim:
            raise ValueError("To compute `features_for_coef`, feature_dim must be larger than output dim")

        feature_dim = int(feature_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        num_groups = max(1, int(num_groups))
        epsilon = float(epsilon)

        if relaxation_factor < 0.:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        if norm_type not in ['batch', 'group']:
            raise ValueError("`norm_type` must be either `batch` or `group`")

        self.feature_columns = feature_columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.norm_type = norm_type
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon

        if num_decision_steps > 1:
            features_for_coeff = feature_dim - output_dim
            print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")

        if self.feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)

            if self.norm_type == 'batch':
                self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum, name='input_bn')
            else:
                self.input_bn = GroupNormalization(axis=-1, groups=self.num_groups, name='input_gn')

        else:
            self.input_features = None
            self.input_bn = None

        self.transform_f1 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f1')

        self.transform_f2 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f2')

        self.transform_f3_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f3_{i}')
            for i in range(self.num_decision_steps)
        ]

        self.transform_f4_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f4_{i}')
            for i in range(self.num_decision_steps)
        ]

        self.transform_coef_list = [
            TransformBlock(self.num_features, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'coef_{i}')
            for i in range(self.num_decision_steps - 1)
        ]

        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)

        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.

        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below.=
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) +
                            transform_f1) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * tf.math.sqrt(0.5)

            if (ni > 0 or self.num_decision_steps == 1):
                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)

                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = transform_f4[:, self.output_dim:]

            if ni < (self.num_decision_steps - 1):
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = self.transform_coef_list[ni](features_for_coef, training=training)
                mask_values *= complementary_aggregated_mask_values
                mask_values = sparsemax(mask_values, axis=-1)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon), axis=1)) / (
                                     tf.cast(self.num_decision_steps - 1, tf.float32))

                # Add entropy loss
                entropy_loss = total_entropy

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

                # Visualization of the feature selection mask at decision step ni
                # tf.summary.image(
                #     "Mask for step" + str(ni),
                #     tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                #     max_outputs=1)
                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                # This branch is needed for correct compilation by tf.autograph
                entropy_loss = 0.

        # Adds the loss automatically
        self.add_loss(self.sparsity_coefficient * entropy_loss)

        # Visualization of the aggregated feature importances
        # tf.summary.image(
        #     "Aggregated mask",
        #     tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
        #     max_outputs=1)

        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask

        return output_aggregated

    @property
    def feature_selection_masks(self):
        return self._step_feature_selection_masks

    @property
    def aggregate_feature_selection_mask(self):
        return self._step_aggregate_feature_selection_mask


class TabNetClassifier(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_classes,
                 num_features=None,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=1,
                 epsilon=1e-5,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_classes: Number of classes.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'group' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.tabnet = TabNet(feature_columns=feature_columns,
                             num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             **kwargs)

        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False, name='classifier')

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.clf(self.activations)

        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)


class TabNetRegressor(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_regressors,
                 num_features=None,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=1,
                 epsilon=1e-5,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_regressors: Number of regression variables.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'group' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNetRegressor, self).__init__(**kwargs)

        self.num_regressors = num_regressors

        self.tabnet = TabNet(feature_columns=feature_columns,
                             num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             **kwargs)

        self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False, name='regressor')

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.regressor(self.activations)
        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)


# Aliases
TabNetClassification = TabNetClassifier
TabNetRegression = TabNetRegressor
