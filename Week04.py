from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas


def calc_entropy(column):
    if column != 0:
        return -column * np.log2(column)
    else:
        return 0


def calc_STotal(classes, target):
    S = 0
    targets = np.zeros(max(classes))
    for datam in target:
        targets[datam-1] += 1
    for clas in classes:
        S += calc_entropy(targets[clas-1]/target.shape[0])
    return S


def calc_SFreq(S, FClasses, TClasses, data, target):
    Gain = np.zeros(data.shape[1])
    for i in range(0,data.shape[1]):
        nums = np.zeros((FClasses[i].shape[0], TClasses.shape[0]))
        Freq = np.zeros(FClasses[i].shape[0])
        for index in range(0,data.shape[0]):
            nums[data[index][i] - 1][target[index] - 1] += 1
            Freq[data[index][i] - 1] += 1
        Gain[i] = S
        num = 0
        for index in range(0,nums.shape[0]):
            for nindex in range(0,nums.shape[1]):
                num += calc_entropy(nums[index][nindex]/Freq[index])
            Gain[i] -= (Freq[index]/data.shape[0]) * num
            num = 0
    return Gain


def make_tree(data, target, FeaturesLeft):
    targetclass = np.unique(target)
    if targetclass.shape[0] == 1:
        return targetclass[0]
    elif FeaturesLeft.size == 0:
        nums = np.zeros(max(target))
        for t in target:
            nums[t-1] += 1
        return np.argmax(nums)


class IC3Classifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        tree = data, target
        Tclasses = np.unique(target)
        FClasses = np.zeros(data.shape[1], dtype=object)
        for i in range(0, data.shape[1]):
            FClasses[i] = np.unique(data[:,i])
        S = calc_STotal(Tclasses, target)
        SFreq = calc_SFreq(S, FClasses, Tclasses, data, target)

        return IC3Model(tree)


class IC3Model:
    def __init__(self, tree):
        self.tree = tree


    def predict(self, data):
        pass


def main():
    rownums, rows, targets = np.split(pandas.read_csv("C:\\Users\\Brad Borget\\Documents\\lenses.data",
                                                      delim_whitespace=True, header=None).as_matrix(), [1, 5], axis=1)
    train_data, test_data, train_target, test_target = train_test_split(rows, targets)
    clf = IC3Classifier()
    clf.fit(train_data, train_target)




if __name__ == "__main__":
    main()




