from sklearn import datasets

iris_dataset = datasets.load_iris()

#print(iris_dataset.keys()) # Key of table
#print(iris_dataset['DESCR']) # Description
print(iris_dataset['target'])
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(iris_dataset['data'][:10])