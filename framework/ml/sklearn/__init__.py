import sklearn
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
import numpy as np

'''
learning material : https://www.cnblogs.com/wj-1314/p/10179741.html
api : https://scikit-learn.org/stable/modules/classes.html
'''

'dataset'
# use existing dataset
digits = datasets.load_digits(return_X_y=False)
print(digits)
print((digits.data.shape, digits.target.shape, digits.images.shape))

# # make dataset
data, label = datasets.make_classification(n_samples=100, n_features=10, n_classes=2)  # classification
data, label = datasets.make_blobs(n_samples=100,n_features=2,centers=3,cluster_std=[1.0,2.0,3.0])  # clustering

'preprocessing'
# standard scaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
# fit : return scaler, not scale data
# fit_transform : return scaler, scale data
# transform : not return scaler, scale data
# always, we scale train_data and test_data simultaneously
# if cannot, we must scale them by the same scaler
scaler = preprocessing.StandardScaler().fit(data)
scaler.transform(data)
preprocessing.scale(data)  # same thing

# min-max scaler
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
scaler.transform(data)
preprocessing.minmax_scale(data)

# normalization
data_normalized = preprocessing.normalize(data, norm='l2')

# one-hot
data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]  # four samples, three features
print(data)
enc = preprocessing.OneHotEncoder().fit(data)  # see https://www.cnblogs.com/zhoukui/p/9159909.html
print(enc.transform([[0, 1, 3]]).toarray())  # if there is not toarray(), it will return sparse form, we can use sparse=False

# binarization
binarizer = preprocessing.Binarizer(threshold=0.5)
binarizer.transform(data)

# label encoding
le = preprocessing.LabelEncoder()
le.fit(["paris","paris","tokyo","amsterdam"])
le.transform(["tokyo","tokyo","paris"])

'model'
# linear regression
from sklearn.linear_model import LinearRegression
model_liear_reg = LinearRegression(fit_intercept=True, normalize=False,
    copy_X=True, n_jobs=1)

# logistic regression
from sklearn.linear_model import LogisticRegression
model_logit_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
    fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    verbose=0, warm_start=False, n_jobs=1)

# naive bayes
from sklearn import naive_bayes
model = naive_bayes.GaussianNB()
model = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

# decision tree
from sklearn import tree
# also DecisionTreeRegression
model =tree.DecisionTreeClassifier(criterion='gini', max_depth=None,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features=None, random_state=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
     class_weight=None, presort=False)

# svm
from sklearn.svm import SVC  # SVR: regression
model = SVC(C=1.0, kernel='rbf', gamma='auto')

# knn
from sklearn import neighbors
model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)

# neural network
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)

'evaluation'
# k-fold
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
Y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=5)
print(kf.get_n_splits(X))
model = sklearn.linear_model.LinearRegression()
data, label = pd.DataFrame(X), pd.DataFrame(Y)
for train_index, test_index in kf.split(X):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    y_train, y_test = label.iloc[train_index], label.iloc[test_index]
    model.fit(X_train, y_train)

# leave one out
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.split(X)

# cross validation using k-fold
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data, label, scoring='accuracy', cv=5, n_jobs=1)  # cv: 5-Fold
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  # Accuracy: 0.98 (+/- 0.03)

# cross validation using stratified k-fold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
strKf = StratifiedKFold(n_splits=3,shuffle=False,random_state=0)
scores = cross_val_score(model, data, label, scoring='accuracy', cv=strKf, n_jobs=1)  # cv: 5-Fold

# also, cv=loo, cv=shufflesplit

# validation curve
from sklearn.model_selection import validation_curve
train_score, test_score = validation_curve(model, X, Y, param_name='a', param_range=(0, 1), cv=None, scoring=None, n_jobs=1)

# classification
import sklearn.metrics as metrics
metrics.accuracy_score(y_true=Y, y_pred=None)
print(metrics.classification_report(
    y_true=None, y_pred=None
))
print(metrics.confusion_matrix(None, None))
# precision_score, recall_score, f1_score: average='micro'/'macro'/'weighted'
# roc_curve, roc_auc

'io'
# pickle
import pickle
with open('model.pickle','wb')as f:
    pickle.dump(model, f)
with open('model.pickle','rb')as f:
    model = pickle.load(f)

# joblib
from sklearn.externals import joblib
joblib.dump(model,'model.pickle')
model = joblib.load('model.pickle')

