import numpy as np
import perceptron
import datagens

NUM_EXAMPLES = 100000
NUM_TEST = 1000

def main():
    training_x, training_y = training_data = datagens.logical_and(size=NUM_EXAMPLES)
    test_x, test_y = test_data = datagens.logical_and(size=NUM_TEST)
    model = perceptron.Perceptron(training_data)
    predictions = model.predictions(test_x)
    temp = predictions - test_y
    temp[temp != 0] = 1
    accuracy = np.sum(temp) / NUM_TEST
    print("Number of examples: {}".format(NUM_EXAMPLES))
    print("Number of test points: {}".format(NUM_TEST))
    print("Accuracy of model: {}".format(accuracy))

if __name__ == "__main__":
    main()
