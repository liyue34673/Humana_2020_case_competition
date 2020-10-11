# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_cell_magic('time', '', "train_data = pd.read_csv('../data/cleaned_train.csv')\n# train_data = train_data.drop(columns=['person_id_syn'])")

y = train_data.transportation_issues
X = train_data.drop(columns=['transportation_issues', 'person_id_syn'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2020)

clf = RandomForestClassifier(max_depth=50, criterion='gini', random_state=2020, class_weight='balanced', n_jobs=-1)
clf.fit(X_train, y_train)

y_hat = clf.predict(X_train)

print(classification_report(y_train, y_hat))

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

pd.Series(y_pred).value_counts()


y_test.value_counts()

feature_importance = pd.DataFrame({'feature':X.columns, 'importance':clf.feature_importances_}).sort_values('importance',ascending=False).reset_index().drop(columns='index')
fig, ax = plt.subplots()
fig.set_size_inches(8.27,15)
plt.title('Feature Importance Plot')
sns.barplot(x='importance',y='feature',ax=ax,data=feature_importance[:50])

print("test LogLoss", round(log_loss(y_test, y_pred), 4))
print("test AUC", round(roc_auc_score(y_test, y_pred), 4))
