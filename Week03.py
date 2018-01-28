from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas
import math


UciHeaders = ["Buying", "Maintenance", "Doors", "Persons", "Lug", "Safety", "Target"]
PimaHeaders = ["Pregnant", "Plasma", "Blood Pressure", "Triceps", "Insulin", "BMI", "DPF", "Age", "Target"]
MPGHeaders = ["Target", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year",
              "Origin", "Name"]


def UCInorm(data):

    data["Buying"] = data["Buying"].astype('category').cat.codes
    data["Maintenance"] = data["Maintenance"].astype('category').cat.codes
    data["Doors"] = data["Doors"].astype('category').cat.codes
    data["Persons"] = data["Persons"].astype('category').cat.codes
    data["Lug"] = data["Lug"].astype('category').cat.codes
    data["Safety"] = data["Safety"].astype('category').cat.codes
    data["Target"] = data["Target"].astype('category').cat.codes
    return data


def Pimanorm(data):
    Pregnant = data["Pregnant"].median()
    Plasma = data["Plasma"].median()
    BP = data["Blood Pressure"].median()
    Triceps = data["Triceps"].median()
    Insulin = data["Insulin"].median()
    BMI = data["BMI"].median()
    DPF = data["DPF"].median()
    Age = data["Age"].median()
    data["Pregnant"] = [Pregnant if num == 1 else num for num in data["Pregnant"]]
    data["Plasma"] = [Plasma if num == 1 else num for num in data["Plasma"]]
    data["Blood Pressure"] = [BP if num == 1 else num for num in data["Blood Pressure"]]
    data["Triceps"] = [Triceps if num == 1 else num for num in data["Triceps"]]
    data["Insulin"] = [Insulin if num == 1 else num for num in data["Insulin"]]
    data["BMI"] = [BMI if num == 1 else num for num in data["BMI"]]
    data["DPF"] = [DPF if num == 1 else num for num in data["DPF"]]
    data["Age"] = [Age if num == 1 else num for num in data["Age"]]
    data["Target"] = data["Target"].astype('category').cat.codes
    return data


def Automobilenorm(data):
    data["Name"] = data["Name"].astype('category').cat.codes
    HP = data["Horsepower"].mode()[0]
    data["Horsepower"] = data["Horsepower"].apply(lambda v: v.replace("?", HP))
    data["Horsepower"] = pandas.to_numeric(data["Horsepower"])
    data["Target"] = data["Target"].astype('category').cat.codes
    return data


class KNNClassifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        return KNNModel(data, target)


class KNNModel:
    def __init__(self, data, target):
        self.data = data.as_matrix()
        self.target = target.as_matrix()

    def predict(self, k, data):
        data = data.as_matrix()
        closest = []
        for row in data:
            distances = []
            for srow in self.data:
                distance = 0
                for j in range(0, srow.shape[0]):
                    distance += (row[j] - srow[j]) ** 2
                distances.append(distance)
            index = np.argsort(distances, axis=0)
            classes = np.unique(self.target[index[:k]])
            if len(classes) == 1:
                closest.append(int(classes[0]))
            else:
                countfreqclasses = np.zeros(int(max(classes)) + 1)
                for j in range(k):
                    countfreqclasses[int(self.target[index[j]])] += 1
                closest.append(np.argmax(countfreqclasses))
        return closest


def main():

    # read in the data
    UCI = pandas.read_csv("C:\\Users\\Brad Borget\\Documents\\car.data",
                          names=UciHeaders, header=None)
    Pima = pandas.read_csv("C:\\Users\\Brad Borget\\Documents\\pima-indians-diabetes.data",
                           names=PimaHeaders, header=None)
    Automobile = pandas.read_csv("C:\\Users\\Brad Borget\\Documents\\auto-mpg.data",
                                 names=MPGHeaders, header=None, delim_whitespace=True)

    # normalize the data
    UCI = UCInorm(UCI)
    Pima = Pimanorm(Pima)
    Automobile = Automobilenorm(Automobile)
    kf = KFold(n_splits=5, shuffle=True)
    clf = KNNClassifier()
    classifier = KNeighborsClassifier(n_neighbors=4)
    data, target = np.split(UCI, [6], axis=1)
    # Test the data
    print("UCI data set:")
    for train_index, test_index in kf.split(UCI):
        train_data, test_data = data.loc[train_index, :], data.loc[test_index, :]
        train_target, test_target = target.loc[train_index, :], target.loc[test_index, :]
        model = clf.fit(train_data, train_target)
        targetspredicted = model.predict(4, test_data)
        model = classifier.fit(train_data, train_target)
        predictions = model.predict(test_data)
        i = 0
        test_target = test_target.as_matrix()
        for k, j in zip(targetspredicted, test_target):
            if k == j:
                i += 1
        num = test_target.shape[0]

        i = (100 * i) / num
        print(str(i) + "% accuracy (mine)")
        i = 0
        for test in zip(predictions, test_target):
            k, j = test
           if k == j:
                i += 1
        i = (100 * i) / num
        print(str(i) + "% accuracy (theirs)")
    target, data = np.split(Automobile, [1], axis=1)
    print("Automobile data set:")
    for train_index, test_index in kf.split(Automobile):
        train_data, test_data = data.loc[train_index, :], data.loc[test_index, :]
        train_target, test_target = target.loc[train_index, :], target.loc[test_index, :]
        model = clf.fit(train_data, train_target)
        targetspredicted = model.predict(4, test_data)
        model = classifier.fit(train_data, train_target)
        predictions = model.predict(test_data)
        i = 0
        test_target = test_target.as_matrix()
        for k, j in zip(targetspredicted, test_target):
            if k == j:
                i += 1
        num = test_target.shape[0]
        i = (100 * i) / num
        print(str(i) + "% accuracy (mine)")
        i = 0
        for test in zip(predictions, test_target):
            k, j = test
            if k == j:
                i += 1
        i = (100 * i) / num
        print(str(i) + "% accuracy (theirs)")
    data, target = np.split(Pima, [8], axis=1)
    print("Pima data set:")
    for train_index, test_index in kf.split(Pima):
        train_data, test_data = data.loc[train_index, :], data.loc[test_index, :]
        train_target, test_target = target.loc[train_index, :], target.loc[test_index, :]
        model = clf.fit(train_data, train_target)
        targetspredicted = model.predict(4, test_data)
        model = classifier.fit(train_data, train_target)
        predictions = model.predict(test_data)
        i = 0
        test_target = test_target.as_matrix()
        for test in zip(targetspredicted, test_target):
            k, j = test
            if k == j:
                i += 1
        num = test_target.shape[0]
        i = (100 * i) / num
        print(str(i) + "% accuracy (mine)")
        i = 0
        for test in zip(predictions, test_target):
            k, j = test
            if k == j:
                i += 1
        i = (100 * i) / num
        print(str(i) + "% accuracy (theirs)")


if __name__ == "__main__":
    main()