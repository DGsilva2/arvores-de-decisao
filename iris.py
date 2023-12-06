from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = load_iris()
iris.keys() #quais sao as chaves do dicionario 

x = iris.data[:, 2:]
y = iris['target']

tree_clf = DecisionTreeClassifier()
tree_clf.fit(x, y)

fig, ax= plt.subplots(figsize=(15,8))
tree.plot_tree(tree_clf)

from sklearn.datasets import make_blobs
plt.style.use('ggplot')
x,y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0, cluster_std=2)

plt.scatter(x[:,0 ], x[:, 1], c=y)

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x, y)

x_min, x_max = x[:, 0].min(), x[:, 0].max()
y_min, y_max = x[:, 1].min(), x[:, 1].max()

xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1), np.arange(y_min,y_max, 0.1))

fig, ax= plt.subplots(sharex='col', sharey='row', figsize=(10,8))

z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

ax.contour(xx, yy, z, alpha=0.5)
ax.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolors='k')
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])

#modelo de regressao
from sklearn.tree import DecisionTreeRegressor
rng = np.random.RandomState(1)
x = np.sort(5*rng.rand(80, 1), axis=0)
y = np.sin(x).ravel() 
y[::5] += 3 * (0.5 - np.random.rand(16))

plt.scatter(x, y)

#fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x, y)
regr_2.fit(x, y)
fig, ax = plt.subplots(figsize=(20, 8))
tree.plot_tree(regr_1)


x_test = np.arange(0, 5, 0.01)[:, np.newaxis]
y_pred_1 = regr_1.predict(x_test)
y_pred_2 = regr_2.predict(x_test)

plt.figure()
plt.scatter(x, y, s=20, edgecolors='black', c='darkorange', label='data')
plt.plot(x_test, y_pred_1, color='cornflowerblue', label='max_depth=2', linewidth=2)
plt.plot(x_test, y_pred_2, color='red', label='max_depth=5', linewidth=2)

plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()

#ENSEBLE LEARNING
from sklearn.datasets import load_breast_cancer
breasr_cancer = load_breast_cancer()
breasr_cancer.keys()
x = breasr_cancer['target_names']
np.array(np.unique(x, return_counts=True)).T# VISUALIZANDO QUANTOS TIPOS DE TUMOR 

breasr_cancer['feature_names']

x = breasr_cancer.data[:, 3:5]
y = breasr_cancer['target']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

tree_clf = DecisionTreeClassifier()
lr_clf = LogisticRegression()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('lr', lr_clf), ('tree', tree_clf), ('svm', svm_clf)], voting='hard')
voting_clf.fit(x_train, y_train)    

from sklearn.metrics import accuracy_score
for clf in (lr_clf, tree_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



x = breasr_cancer.data[:, 0:2]
y = breasr_cancer['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, 
                            max_samples=100, bootstrap=True)

tree_clf = DecisionTreeClassifier(max_depth=40)

bag_clf.fit(x_train, y_train)
tree_clf.fit(x_train, y_train)

#modelo todo confuso 
x_min, x_max = x[:, 0].min(), x[:, 0].max()
y_min, y_max = x[:, 0].min(), x[:, 0].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1 ), np.arange(y_min, y_max, 0.1 ))
fig, ax = plt.subplots(sharex='col', sharey='row', figsize=(10, 8))
z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
ax.contourf(xx, yy, z, alpha=0.8)
ax.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_xlabel(breasr_cancer['feature_names'][0])
ax.set_ylabel(breasr_cancer['feature_names'][2])
plt.show()

#modelo com a melhor decisao
x_min, x_max = x[:, 0].min(), x[:, 0].max()
y_min, y_max = x[:, 0].min(), x[:, 0].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1 ), np.arange(y_min, y_max, 0.1 ))
fig, ax = plt.subplots(sharex='col', sharey='row', figsize=(10, 8))
z = bag_clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
ax.contourf(xx, yy, z, alpha=0.8)
ax.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_xlabel(breasr_cancer['feature_names'][0])
ax.set_ylabel(breasr_cancer['feature_names'][2])
plt.show()

#precisao dos modelos 
for clf in [bag_clf, tree_clf]:
    y_pred =clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))


# FEATURE IMPORTANCE
breasr_cancer = load_breast_cancer()
x = breasr_cancer.data
y = breasr_cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=16)
rnd_clf.fit(x_train, y_train)

rnd_clf.feature_importances_

feature_importances = pd.DataFrame(rnd_clf.feature_importances_,
                                   index=breasr_cancer['feature_names'],
                                   columns= ['importance']).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(20, 8))
feature_importances.plot(kind='barh', ax=ax)