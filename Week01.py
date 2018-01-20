from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np


class HardCodedClassifier:
    def __init__(self):
        pass


    def fit(self, data, target):
        return HardCodedModel()


class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, data):
        newdata = []
        for databit in data:
            newdata.append(0)
        return newdata


def main():
    iris = datasets.load_iris()
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target)
    clf = HardCodedClassifier()
    model = clf.fit(train_data, train_target)
    targetspredicted = model.predict(test_data)
    i = 0
    for test in zip(targetspredicted, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100 * i) / test_target.shape[0]
    print(str(i) + "% accuracy")


if __name__ == "__main__":
    main()
