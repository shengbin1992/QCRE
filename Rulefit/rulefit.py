import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from rulefit import RuleFit


boston_data = pd.read_csv("/Users/binbin/Documents/PhD/Research/Break event/data/Workbook2.csv", index_col=0)

y1 = boston_data.y.values
X1 = boston_data.drop("y", axis=1)
features = X1.columns
X1 = X1.as_matrix()

X_train = X1[:795]
y_train = y1[:795]
X_test = X1[795:]
y_test = y1[795:]

typ='classifier' #regressor or classifier

if typ=='regressor':
    rf = RuleFit(tree_size=4,sample_fract='default',max_rules=2000,
             memory_par=0.01,
             tree_generator=None,
            rfmode='regress',lin_trim_quantile=0.025,
            lin_standardise=True, exp_rand_tree_size=True,random_state=1) 
    rf.fit(X_train, y_train, feature_names=features)
    y_pred=rf.predict(X_test)
    insample_rmse=np.sqrt(np.sum((y_pred-y_train)**2)/len(y_train))
elif typ=='classifier':
    y_class=y_train.copy()
    y_class[y_class<1]=-1
    y_class[y_class>=1]=+1
    N=X.shape[0]
    rf = RuleFit(tree_size=4,sample_fract='default',max_rules=2000,
                 memory_par=0.01,
                 tree_generator=None,
                rfmode='classify',lin_trim_quantile=0.025,
                lin_standardise=True, exp_rand_tree_size=True,random_state=1) 
    rf.fit(X_train, y_class, feature_names=features)
    y_pred=rf.predict(X_test)
    insample_acc=len(y_pred==y_class)/len(y_class)
rules = rf.get_rules()

rules = rules[rules.coef != 0].sort_values(by="support")
num_rules_rule=len(rules[rules.type=='rule'])
num_rules_linear=len(rules[rules.type=='linear'])
print(rules)
