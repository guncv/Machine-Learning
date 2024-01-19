from sklearn import datasets
import matplotlib.pyplot as plt
import pylab

iris_dataset = datasets.load_iris()

#print(iris_dataset.keys()) # Key of table
#print(iris_dataset['DESCR']) # Description
print(iris_dataset['target'])
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(iris_dataset['data'][:10])

digit_dataset = datasets.load_digits()

print(digit_dataset.keys())
print(digit_dataset['images'][:20])
print(digit_dataset['target_names'])
print(digit_dataset.images[0])
pylab.imshow(digit_dataset.images[0],cmap=pylab.cm.gray_r)
pylab.show()