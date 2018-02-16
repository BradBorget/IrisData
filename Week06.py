import random
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas


PimaHeaders = ["Pregnant", "Plasma", "Blood Pressure", "Triceps", "Insulin", "BMI", "DPF", "Age", "Target"]


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


class Node:
    def __init__(self, numWeights):
        self.weights = [];
        for i in range(numWeights + 1):
            self.weights.append(random.uniform(-1, 1))


    def calcInput(self, datam):
        input = 0
        i = 0
        for weight in self.weights:
            if i == 0:
                input += weight * -1
            else:
                input += weight * datam[i-1]
            i += 1
        return input > 0


def createneurons(numNeurons, numWeights):
    NodeList = []
    for i in range(numNeurons):
        NodeList.append(Node(numWeights))
    return NodeList


def main():
    iris = datasets.load_iris()
    Pima = pandas.read_csv("C:\\Users\\Brad Borget\\Documents\\pima-indians-diabetes.data",
                           names=PimaHeaders, header=None)
    Pima = Pimanorm(Pima)
    train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target)
    nodes = createneurons(4, train_data.shape[1])
    value = nodes[0].calcInput(train_data[0])
    if value == True or value == False:
        print("We're GOLDEN!")



if __name__ == "__main__":
    main()