from tabnet.tabnet import TabNet
from tabnet.tabnet import TabNetClassifier
from tabnet.tabnet import TabNetRegressor

from tabnet.tabnet import TabNetClassification
from tabnet.tabnet import TabNetRegression

from tabnet.stacked_tabnet import StackedTabNet
from tabnet.stacked_tabnet import StackedTabNetClassifier
from tabnet.stacked_tabnet import StackedTabNetRegressor

tabnets = ['TabNet', 'TabNetClassifier', 'TabNetRegressor']
stacked_tabnets = ['StackedTabNet', 'StackedTabNetClassifier', 'StackedTabNetRegressor']

__all__ = [*tabnets, *stacked_tabnets]


__version__ = '0.1.6'
