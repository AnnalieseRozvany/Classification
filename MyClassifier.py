import csv
import math
import sys
import re

training = sys.argv[1]
testing = sys.argv[2]
algorithm = sys.argv[3]

numberOfYes = 0
numberOfNo = 0

def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def euclideanDistance(dataA, dataB):
    distance = 0
    for index, i in enumerate(dataA):
        if(len(dataB) <= index):
            return math.sqrt(distance)
        distance += math.pow((i - dataB[index]),2)
    return math.sqrt(distance)


def loadCSV(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        for index, x in enumerate(dataset[i]):
            if (isFloat(x)):
                dataset[i][index] = float(x)
            else:
                dataset[i][index] = x.strip()
    return dataset

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        dataClass = vector[-1]
        vector.remove(dataClass)
        separated[dataClass].append(vector)

    global numberOfYes
    numberOfYes = len(separated["yes"])
    global numberOfNo
    numberOfNo = len(separated["no"])
    return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent #probability density function

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x,mean,stdev)
    probabilities["yes"] *= (float(numberOfYes)/(numberOfYes+numberOfNo))
    probabilities["no"] *= (float(numberOfNo)/(numberOfYes+numberOfNo))
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    if (probabilities["no"] > probabilities["yes"]):
        return "no"
    else:
        return "yes"

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		print(result)

def kNearestNeighbour(k, trainingSet, inputVector):
    neighbours = [(math.inf,0)] * k
    for classValue, instance in trainingSet.items():
        for i in instance:
            distance = euclideanDistance(inputVector, i)
            neighbours.sort()
            if (neighbours[k-1][0] > distance):
                neighbours[k-1] = (distance, classValue)

    no = 0
    yes = 0

    for distance, classValue in neighbours:
        if classValue == "no":
            no += 1
        if classValue == "yes":
            yes+= 1

    if (no > yes):
        return "no"
    elif (yes > no):
        return "yes"
    else:
        return "yes"

def getKNNPredictions(k, trainingSet, testingSet):
    separated = separateByClass(trainingSet)
    for i in range(len(testingSet)):
        result = kNearestNeighbour(k, separated, testingSet[i])
        print(result)


trainingData = loadCSV(training)
testingData = loadCSV(testing)

if (algorithm == "NB"):
    summaries = summarizeByClass(trainingData)
    getPredictions(summaries, testingData)
elif (re.match('^[1-9]NN$',algorithm)):
    getKNNPredictions(int(algorithm[0]), trainingData, testingData)
