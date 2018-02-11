import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class Perceptron(object): # Perceptron class 
    #Initialization
    # c is the learning rate with default learning of 0.01 
    # loops is the number of iteration with default of 40
    def __init__(self, c=0.01, loops=40): 
        self.c = c
        self.loops = loops      

    # Training funcation
    # X is the values to be used form training
    # y is identification of the values -1 or +1
    # Sets the weight values in w_
    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.loops):
            errors = 0
            for xi, target in zip(X, y):
                update = self.c * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors==0 :
                return self
        return self
    # used in prediction
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Predicts type based on w_ (Weigths) 
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def main():
    df = pd.read_csv('train.txt', header=None)      # Read values
    
    y1 = df.iloc[0:df.size, 4].values               
    y1 = np.where(y1 == 'Iris-virginica', -1, 1)    # set to -1 if Iris-virginica otherwise set to +1

    y2 = df.iloc[0:df.size, 4].values
    y2 = np.where(y2 == 'Iris-setosa', -1, 1)       # set to -1 if Iris-setosa otherwise set to +1

    
    X = df.iloc[0:df.size, [0,1,2,3]].values        # Isolate data values

    per1 = Perceptron()                             # Create Perceptron 
    per2 = Perceptron()

    per1.train(X,y1)                                # Train Perceptron
    per2.train(X,y2)
    
    # Graphs showing total errors over time
    plt.title('Perceptron 1 Misclassification Errors')
    plt.plot(range(1, len(per1.errors_)+1), per1.errors_, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassifications')
    plt.show()
    
    plt.title('Perceptron 2 Misclassification Errors')
    plt.plot(range(1, len(per2.errors_)+1), per2.errors_, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Misclassifications')
    plt.show()

    # Running Perceptrons of the teat data
    df2 = pd.read_csv('test.txt', header=None)
    
    X2 = df2.iloc[0:df2.size, [0,1,2,3]].values
    X3 = df2.iloc[0:df2.size, [0,1,2,3,4]].values

    IVir = per1.predict(X2)
    IS = per2.predict(X2)
    Virginica = []
    Setosa = []
    Versicolor = []
    ErrorsTest = 0

    # Seperating points into arrays depending on perceptron predictions 
    for i in range(len(X2)) :
        if IVir[i] == -1 :
            Virginica.append(X3[i])
            if X3[i,4] != 'Iris-virginica':
                ErrorsTest+=1
        elif IS[i] == -1 :
            Setosa.append(X3[i])
            if X3[i,4] != 'Iris-setosa':
                ErrorsTest+=1
        else :
            Versicolor.append(X3[i])
            if X3[i,4] != 'Iris-versicolor':
                ErrorsTest+=1

    f = open('Output.txt','w')
    f.write('Running double perceptron on test data:\n\n')
    for i in Virginica :
        f.write('Point: %s\n'%i[0:3])
        f.write('Flower Type: %s\n'%i[4])
        f.write('Guessed Type: Iris-virginica\n\n')
    for i in Setosa :
        f.write('Point: %s\n'%i[0:3])
        f.write('Flower Type: %s\n'%i[4])
        f.write('Guessed Type: Iris-setosa\n\n')
    for i in Versicolor :
        f.write('Point: %s\n'%i[0:3])
        f.write('Flower Type: %s\n'%i[4])
        f.write('Guessed Type: Iris-versicolor\n\n')

    f.write('Weights:\n\nAll initial weights set to zero\nFinal Weights:\nPerceptron1 Weights: %s\nPerceptron2 Weights: %s\n\n'%(per1.w_, per2.w_))

    f.write('Total Erros: %s\n'%ErrorsTest)

    f.write('\nPerseptron training is set to run for 40 iteration or untill the total errors reach zero\n\n')
    
    f.write('Precision: %s\nRecall: %s\n'%(((len(X2)-ErrorsTest)/len(X2)),((len(X2)-ErrorsTest)/len(X2))))
    

main()
