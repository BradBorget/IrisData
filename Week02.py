from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
        return 0


class KNNClassifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        return KNNModel(data, target)


class KNNModel:
    def __init__(self, data, target):
        self.data = data
        self.target = target


    def predict(self, k, data):
        closest = []
        for row in data:
            distances = []
            for srow in self.data:
                distance = 0
                for j in range(0,srow.shape[0]):
                    distance += (row[j]-srow[j])**2
                distances.append(distance)
            index = np.argsort(distances,axis=0)
            classes = np.unique(self.target[index[:k]])
            if len(classes)==1:
                closest.append(int(classes[0]))
            else:
                countfreqclasses = np.zeros(max(classes)+1)
                for j in range(k):
                    countfreqclasses[self.target[index[:j]]] += 1
                closest.append(np.argmax(countfreqclasses))
        return closest



def main():
    iris = datasets.load_iris()
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target)
    clf = KNNClassifier()
    model = clf.fit(train_data, train_target)
    targetspredicted = model.predict(4, test_data)
    i = 0
    classifier = KNeighborsClassifier(n_neighbors=4)
    model = classifier.fit(train_data, train_target)
    predictions = model.predict(test_data)
    for test in zip(targetspredicted, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100 * i) / test_target.shape[0]
    print(str(i) + "% accuracy (mine)")
    i = 0
    for test in zip(predictions, test_target):
        k, j = test
        if k == j:
            i += 1
    i = (100*i)/test_target.shape[0]
    print(str(i) + "% accuracy (theirs)")



if __name__ == "__main__":
    main()
