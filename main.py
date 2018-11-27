# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def loadCsv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csvReader = reader(file)
		for row in csvReader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def strColToFloat(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def strColToInt(dataset, column):
	classVal = [row[column] for row in dataset]
	unique = set(classVal)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, nFolds):
	dataSplit = list()
	dataCpy = list(dataset)
	foldSize = int(len(dataset) / nFolds)
	for i in range(nFolds):
		fold = list()
		while len(fold) < foldSize:
			index = randrange(len(dataCpy))
			fold.append(dataset_copy.pop(index))
		dataSplit.append(fold)
	return dataSplit

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, nFolds, *args):
	folds = cross_validation_split(dataset, nFolds)
	scores = list()
	for fold in folds:
		trainSet = list(folds)
		trainSet.remove(fold)
		trainSet = sum(trainSet, [])
		testSet = list()
		for row in fold:
			row_copy = list(row)
			testSet.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(trainSet, testSet, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]

	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent

def trainWeights(train, learnRate, nEpoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(nEpoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + learnRate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + learnRate * error * row[i]

	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, learnRate, nEpoch):
	predictions = list()
	weights = trainWeights(train, learnRate, nEpoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)

	return(predictions)

# Test the Perceptron algorithm on the sonar dataset
seed(1)

# load and prepare data

filename = 'Sonar.csv'
dataset = loadCsv(filename)

for i in range(len(dataset[0]) - 1):
	strColToFloat(dataset, i)

# convert string class to integers

strColToInt(dataset, len(dataset[0])-1)

# evaluate algorithm

nFold = 3
learnRate = 0.01
nEpoch = 500
scores = evaluate_algorithm(dataset, perceptron, nFold, learnRate, nEpoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))