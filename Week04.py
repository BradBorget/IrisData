from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas


class Node:
    def __init__(self, val, col, tar):
        self.branches = []
        self.v = val
        self.col = col
        self.tar = tar

    def print(self, tree=None, numTabs=0):
        if tree == None:
            tree = self
        print("Value is: {} Column is: {} Target is: {}".format(tree.v, tree.col, tree.tar))
        if len(tree.branches) > 0:
            for tab in range(numTabs):
                print("\t", end="")
            numTabs += 1;
            print("[")
            for branch in tree.branches:
                for tab in range(numTabs):
                    print("\t", end="")
                self.print(branch, numTabs)
            for tab in range(1, numTabs):
                print("\t", end="")
            print("]")


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
    for i in range(data.shape[1]):
        nums = np.zeros((max(FClasses[i]), max(TClasses)))
        Freq = np.zeros(max(FClasses[i]))
        for index in range(data.shape[0]):
            nums_copy = nums[data[index][i] - 1]
            nums_copy[target[index] - 1] += 1
            Freq[data[index][i] - 1] += 1
        Gain[i] = S
        num = 0
        for index in range(nums.shape[0]):
            for nindex in range(nums.shape[1]):
                num += calc_entropy(nums[index][nindex]/Freq[index])
            Gain[i] -= (Freq[index]/data.shape[0]) * num
            num = 0
    return Gain


def make_tree(data, target, FeaturesLeft, SFreq, datam, index, indices=[]):
    targetclass = np.unique(target)
    if targetclass.shape[0] == 1:
        return Node(datam, index, targetclass[0])
    elif FeaturesLeft.size == 0 or data == []:
        default = target[np.argmax(target)][0][0][0]
        return Node(datam, index, default)
    else:
        if indices == []:
            index = np.argmax(SFreq)
            for i in range(FeaturesLeft.shape[0]):
                if i != index:
                    indices.append(i)
            tree = Node(datam, index, None)
            index2 = index
        else:
            tree = Node(datam, index, None)
            index = np.argmax(SFreq)
            index2 = indices.pop(index)
        col = FeaturesLeft[index]
        FeaturesLeft = np.delete(FeaturesLeft, index, 0)
        for datam in col:
            ixgrid = np.ix_(data[:, index] == datam, indices)
            FClasses = np.zeros(FeaturesLeft.shape[0], dtype=object)
            j = 0
            for i in ixgrid[1][0]:
                FClasses[j] = np.unique(data[:, i])
                j += 1
            S = calc_STotal(np.unique(target[ixgrid[0]]), target[ixgrid[0]])
            SFreq = calc_SFreq(S, FClasses, targetclass, data[ixgrid], target[ixgrid[0]])
            if indices == []:
                tree.branches.append(make_tree([], target[ixgrid[0]], FeaturesLeft, SFreq, datam, index2, indices))
            else:
                tree.branches.append(make_tree(data[ixgrid], target[ixgrid[0]], FeaturesLeft, SFreq, datam, index2, indices))
        return tree


def predictRecurse(data, tree):
    for item in tree.branches:
        if data[item.col] == item.v and len(item.branches) == 0:
            return item.tar
        elif data[item.col] == item.v:
            return predictRecurse(data, item)


class IC3Classifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        Tclasses = np.unique(target)
        FClasses = np.zeros(data.shape[1], dtype=object)
        for i in range(0, data.shape[1]):
            FClasses[i] = np.unique(data[:,i])
        S = calc_STotal(Tclasses, target)
        SFreq = calc_SFreq(S, FClasses, Tclasses, data, target)
        tree = make_tree(data, target, FClasses, SFreq, 0, 0)
        tree.print()
        return IC3Model(tree)


class IC3Model:
    def __init__(self, tree):
        self.tree = tree


    def predict(self, data):
        targets = []
        for datam in data:
            print(datam)
            targets.append(predictRecurse(datam, self.tree))
        return targets


def main():
    rownums, rows, targets = np.split(pandas.read_csv("C:\\Users\\Brad Borget\\Documents\\lenses.data",
                                                      delim_whitespace=True, header=None).as_matrix(), [1, 5], axis=1)
    clf = IC3Classifier()
    model = clf.fit(rows, targets)
    train_data, test_data, train_target, test_target = train_test_split(rows, targets)
    targets = model.predict(test_data)
    j = 0
    for i in range(len(targets)):
        if test_target[i] == targets[i]:
            j += 1
    percentage = (j * 100)/len(targets)
    print("Ours: {}%".format(percentage))




if __name__ == "__main__":
    main()
