import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from rulefit import RuleFit


boston_data = pd.read_csv("/Users/binbin/Documents/PhD/Research/Break event/data/Workbook2.csv", index_col=0)

y1 = boston_data.y.values
X1 = boston_data.drop("y", axis=1)
features = X1.columns
X1 = X1.as_matrix()

X_train = X1[:618]
y_train = y1[:618]
X_test = X1[618:]
y_test = y1[618:]

import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

clf = svm.SVC(kernel='linear')

# Train classifier 
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred)

# Calculate the AUC
roc_auc = auc(fpr, tpr)
print 'ROC AUC: %0.2f' % roc_auc

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
