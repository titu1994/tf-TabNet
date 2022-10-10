import tensorflow as tf
from tabnet.tabnet import TabNet


class StackedTabNet(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_layers=1,
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
                 random_state=None,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.

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
            num_layers: Number of TabNets to stack together.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer. Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
                Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
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
        super(StackedTabNet, self).__init__(**kwargs)

        if num_layers < 1:
            raise ValueError("`num_layers` cannot be less than 1")

        if type(feature_dim) not in [list, tuple]:
            feature_dim = [feature_dim] * num_layers

        if type(output_dim) not in [list, tuple]:
            output_dim = [output_dim] * num_layers

        if len(feature_dim) != num_layers:
            raise ValueError("`feature_dim` must be a list of length `num_layers`")

        if len(output_dim) != num_layers:
            raise ValueError("`output_dim` must be a list of length `num_layers`")

        self.num_layers = num_layers

        layers = []
        layers.append(TabNet(feature_columns=feature_columns,
                             num_features=num_features,
                             feature_dim=feature_dim[0],
                             output_dim=output_dim[0],
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             random_state=random_state))

        for layer_idx in range(1, num_layers):
            layers.append(TabNet(feature_columns=None,
                                 num_features=output_dim[layer_idx - 1],
                                 feature_dim=feature_dim[layer_idx],
                                 output_dim=output_dim[layer_idx],
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 norm_type=norm_type,
                                 batch_momentum=batch_momentum,
                                 virtual_batch_size=virtual_batch_size,
                                 num_groups=num_groups,
                                 epsilon=epsilon,
                                 random_state=random_state))

        self.tabnet_layers = layers

    def call(self, inputs, training=None):
        x = self.tabnet_layers[0](inputs, training=training)

        for layer_idx in range(1, self.num_layers):
            x = self.tabnet_layers[layer_idx](x, training=training)

        return x

    @property
    def tabnets(self):
        return self.tabnet_layers

    @property
    def feature_selection_masks(self):
        return [tabnet.feature_selection_masks
                for tabnet in self.tabnet_layers]

    @property
    def aggregate_feature_selection_mask(self):
        return [tabnet.aggregate_feature_selection_mask
                for tabnet in self.tabnet_layers]


class StackedTabNetClassifier(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_classes,
                 num_layers=1,
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
                 random_state=random_state,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.

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
            num_layers: Number of TabNets to stack together.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer. Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
                Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
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
        super(StackedTabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                            num_layers=num_layers,
                                            feature_dim=feature_dim,
                                            output_dim=output_dim,
                                            num_features=num_features,
                                            num_decision_steps=num_decision_steps,
                                            relaxation_factor=relaxation_factor,
                                            sparsity_coefficient=sparsity_coefficient,
                                            norm_type=norm_type,
                                            batch_momentum=batch_momentum,
                                            virtual_batch_size=virtual_batch_size,
                                            num_groups=num_groups,
                                            epsilon=epsilon,
                                            random_state=random_state)

        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)

    def call(self, inputs, training=None):
        self.activations = self.stacked_tabnet(inputs, training=training)
        out = self.clf(self.activations)

        return out


class StackedTabNetRegressor(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_regressors,
                 num_layers=1,
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
                 random_state=random_state,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
        Stacked variant of the TabNet model, which stacks multiple TabNets into a singular model.

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
            num_regressors: Number of regressors.
            num_layers: Number of TabNets to stack together.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer. Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
                Can be either a single int, or a list of
                integers. If a list, must be of same length as the number of layers.
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
        super(StackedTabNetRegressor, self).__init__(**kwargs)

        self.num_regressors = num_regressors

        self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                            num_layers=num_layers,
                                            feature_dim=feature_dim,
                                            output_dim=output_dim,
                                            num_features=num_features,
                                            num_decision_steps=num_decision_steps,
                                            relaxation_factor=relaxation_factor,
                                            sparsity_coefficient=sparsity_coefficient,
                                            norm_type=norm_type,
                                            batch_momentum=batch_momentum,
                                            virtual_batch_size=virtual_batch_size,
                                            num_groups=num_groups,
                                            epsilon=epsilon,
                                            random_state=random_state)

        self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False)

    def call(self, inputs, training=None):
        self.activations = self.stacked_tabnet(inputs, training=training)
        out = self.regressor(self.activations)
        return out
