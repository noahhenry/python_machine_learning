# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Dimensions of Dataset
# shape
print(dataset.shape) # -> (150, 5)

# Peek at the data
# It is always a good idea to see your data...
# head
print(dataset.head(20)) # -> returns a table of the first 20 rows, see below
#     sepal-length  sepal-width  petal-length  petal-width        class
# 0            5.1          3.5           1.4          0.2  Iris-setosa
# 1            4.9          3.0           1.4          0.2  Iris-setosa
# 2            4.7          3.2           1.3          0.2  Iris-setosa
# 3            4.6          3.1           1.5          0.2  Iris-setosa
# 4            5.0          3.6           1.4          0.2  Iris-setosa
# 5            5.4          3.9           1.7          0.4  Iris-setosa
# 6            4.6          3.4           1.4          0.3  Iris-setosa
# 7            5.0          3.4           1.5          0.2  Iris-setosa
# 8            4.4          2.9           1.4          0.2  Iris-setosa
# 9            4.9          3.1           1.5          0.1  Iris-setosa
# 10           5.4          3.7           1.5          0.2  Iris-setosa
# 11           4.8          3.4           1.6          0.2  Iris-setosa
# 12           4.8          3.0           1.4          0.1  Iris-setosa
# 13           4.3          3.0           1.1          0.1  Iris-setosa
# 14           5.8          4.0           1.2          0.2  Iris-setosa
# 15           5.7          4.4           1.5          0.4  Iris-setosa
# 16           5.4          3.9           1.3          0.4  Iris-setosa
# 17           5.1          3.5           1.4          0.3  Iris-setosa
# 18           5.7          3.8           1.7          0.3  Iris-setosa
# 19           5.1          3.8           1.5          0.3  Iris-setosa

# Satistical Summary
# descriptions
print(dataset.describe()) # -> returns a table with information, see below
#        sepal-length  sepal-width  petal-length  petal-width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

# Class Distribution
# class distribution
print(dataset.groupby('class').size()) # -> returns...
# class
# Iris-setosa        50
# Iris-versicolor    50
# Iris-virginica     50
# dtype: int64

# Complete Example:
# summarize the data
from pandas import read_csv
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())




# 4. Data Visualization

# 4.1 Univariate Plots
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# 4.2 Multivariate Plots
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# 4.3 Complete Example
# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()




# 5 Evaluate Some Algorithms
# Now it is time to create some models of the data and estimate thier accuracy on unseen data.

# 5.1 Create a Validation Dataset
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# 5.2 Test Harness

# 5.3 Build Models
# six different algorithms
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())) # -> returns...
    # LR: 0.941667 (0.065085)
    # LDA: 0.975000 (0.038188)
    # KNN: 0.958333 (0.041667)
    # CART: 0.933333 (0.050000)
    # NB: 0.950000 (0.055277)
    # SVM: 0.983333 (0.033333)

# 5.4 Select Best Model
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()




# 6 Make Predictions

# 6.1 Make Predictions
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# 6.2 Evaluate Predictions
# Evaluate Predictions
print(accuracy_score(Y_validation, predictions))
  # 0.9666666666666667
print(confusion_matrix(Y_validation, predictions))
  # [[11  0  0]
  #  [ 0 12  1]
  #  [ 0  0  6]]
print(classification_report(Y_validation, predictions))
  #                  precision    recall  f1-score   support

  #     Iris-setosa       1.00      1.00      1.00        11
  # Iris-versicolor       1.00      0.92      0.96        13
  #  Iris-virginica       0.86      1.00      0.92         6

  #        accuracy                           0.97        30
  #       macro avg       0.95      0.97      0.96        30
  #    weighted avg       0.97      0.97      0.97        30
