from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
here is the simplest pipeline.
one more complete pipeline is shown in snippet.txt.
you can set the snippet in your IDE.
'''

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.3)

model_knn = KNeighborsClassifier()
model_knn.fit(x_train, y_train)
params = model_knn.get_params()
print(params)

print(model_knn.predict(x_test))
print(model_knn.score(x_test, y_test))


