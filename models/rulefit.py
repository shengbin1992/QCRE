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

typ='classifier' #regressor or classifier

if typ=='regressor':
    rf = RuleFit(tree_size=4,sample_fract='default',max_rules=2000,
             memory_par=0.01,
             tree_generator=None,
            rfmode='regress',lin_trim_quantile=0.025,
            lin_standardise=True, exp_rand_tree_size=True,random_state=1) 
    
    rf.fit(X_train, y_train, feature_names=features)
    y_pred=rf.predict(X_test)
    insample_rmse=np.sqrt(np.sum((y_pred-y_test)**2)/len(y_test))
    
elif typ=='classifier':
    y_class=y_train.copy()
    y_class[y_class<1]=-1
    y_class[y_class>=1]=+1
    N=X_train.shape[0]
    rf = RuleFit(tree_size=4,sample_fract='default',max_rules=2000,
                 memory_par=0.01,
                 tree_generator=None,
                rfmode='classify',lin_trim_quantile=0.025,
                lin_standardise=True, exp_rand_tree_size=True,random_state=1) 
    rf.fit(X_train, y_class, feature_names=features)
    y_pred=rf.predict(X_test)
    y_class1=y_test.copy()
    y_class1[y_class1<1]=-1
    y_class1[y_class1>=1]=+1
    insample_acc=len(y_pred==y_class1)/len(y_class1)
rules = rf.get_rules()

rules = rules[rules.coef != 0].sort_values(by="support")
num_rules_rule=len(rules[rules.type=='rule'])
num_rules_linear=len(rules[rules.type=='linear'])
print(rules)

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
