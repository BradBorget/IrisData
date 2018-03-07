import random
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas
import numpy as np
import math
from sklearn.preprocessing import normalize


PimaHeaders = ["Pregnant", "Plasma", "Blood Pressure", "Triceps", "Insulin", "BMI", "DPF", "Age", "Target"]
LEARNING_RATE = .1

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
        self.weights = []
        for i in range(numWeights + 1):
            self.weights.append(random.uniform(-1, 1))
        self.input = 0
        self.Bj = 0

    def calcInput(self, datam):
        input = 0
        i = 0
        for weight in self.weights:
            if i == 0:
                input += weight * -1
            else:
                input += weight * datam[i-1]
            i += 1
        self.input = 1 / (1 + math.exp(input))

    def getInput (self):
        return self.input

    def updateWeights (self, input, weightnum):
        self.weights[weightnum] = self.weights[weightnum] - (LEARNING_RATE * self.Bj * input)


def createneurons(numNeurons, numWeights):
    NodeList = []
    for i in range(numNeurons):
        NodeList.append(Node(numWeights))
    return np.asarray(NodeList)


def createLayers(numLayers, numNodesPerLayer, numWeightsPerNode):
    layers = []
    for i in range(numLayers):
        layers.append(createneurons(numNodesPerLayer[i], numWeightsPerNode))
        numWeightsPerNode = numNodesPerLayer[i]
    return np.asarray(layers)


def outputUpdate(node, target):
    node.Bj = node.getInput() * (1 - node.getInput()) * (node.getInput() - target)


def hiddenUpdate(node, outputs, weightnum):
    node.Bj = node.getInput() * (1 - node.getInput())
    sum = 0
    for output in outputs:
        sum += output.Bj * output.weights[weightnum]
    node.Bj = node.Bj * sum


class NeuralNetClass:
    def __init__(self):
        pass

    def fit(self, data, target, numHiddenLayers, numNodesPerLayer, numIterations):
        numNodesPerLayer.append(np.unique(target).shape[0])
        layers = createLayers(numHiddenLayers + 1, numNodesPerLayer, data.shape[1])
        for n in range(numIterations):
            targets = []
            h = 0
            for datam in data:
                i = 0
                for layer in layers:
                    for node in layer:
                        if np.array_equal(layer, layers[0]):
                            node.calcInput(datam)
                        else:
                            vals = np.zeros(layers[i-1].shape[0])
                            j = 0
                            for newNode in layers[i-1]:
                                vals[j] = newNode.getInput()
                                j += 1
                            node.calcInput(vals)
                    i += 1
                targets.append(np.argmin(vals))
                #If the guess is wrong.
                if np.argmax(vals) != target[h]:
                    #Get the Bj
                    for i, layer in reversed(list(enumerate(layers))):
                        for j, node in enumerate(layer):
                            if np.array_equal(layer, layers[-1]):
                                outputUpdate(node, 1 if target[h] == j else 0)
                            else:
                                hiddenUpdate(node, layers[i + 1], j+1)
                    #Update the Nodes based on Bj
                    for i, layer in enumerate(layers):
                        for node in layer:
                            if np.array_equal(layer, layers[0]):
                                for j, weight in enumerate(node.weights):
                                    if j == 0:
                                        node.updateWeights(-1, j)
                                        j+=1
                                    node.updateWeights(datam[j-1], j)
                            else:
                                for j, newNode in enumerate(layers[i-1]):
                                    if j == 0:
                                        node.updateWeights(-1, j)
                                        j+=1
                                    node.updateWeights(newNode.getInput(), j)
                h += 1
            print(targets)
        return NeuralNetModel(layers)


class NeuralNetModel:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, data):
        targets = np.zeros(data.shape[0])
        h = 0
        vals = []
        for datam in data:
            i = 0
            for layer in self.layers:
                for node in layer:
                    if np.array_equal(layer, self.layers[0]):
                        node.calcInput(datam)
                    else:
                        vals = np.zeros(self.layers[i-1].shape[0])
                        j = 0
                        for newNode in self.layers[i-1]:
                            vals[j] = newNode.getInput()
                            j += 1
                        node.calcInput(vals)
                i += 1
            targets[h] = np.argmax(vals)
            h += 1
        return targets


def main():
    iris = datasets.load_iris()
    Pima = pandas.read_csv("C:\\Users\\Brad Borget\\Documents\\pima-indians-diabetes.data",
                           names=PimaHeaders, header=None)
    Pima = Pimanorm(Pima).values
    norm2 = normalize(Pima)
    data = norm2[:,8:]
    target = norm2[:,-1:]
    train_data, test_data, train_target, test_target = train_test_split(data, target)
    #train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target)
    NNClass = NeuralNetClass()
    NNModel = NNClass.fit(train_data, train_target, 1, [3], 20)
    targets = NNModel.predict(test_data)


if __name__ == "__main__":
    main()