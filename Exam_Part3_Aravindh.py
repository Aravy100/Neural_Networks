# CSCI 5922: Exam 1 Part 3
# WARNING: PLEASE CLEAR EXISTING VARIABLES IN SPYDER3 BEFORE RUNNING THIS SCRIPT
# The following Python3 code takes an input dataset splits it into train and test, runs it through a single hidden layer
# multiple output neural network and spits out the loss, actual prediction and a confusion matrix.

# Note: '#' (single hash) indicates my comments
# '##' (double hash) are PYTHON CODE STATEMENTS that have been commented out


# Importing the required packages
import numpy as np
import pandas as pd



#-----------------------------------------------------------------------------#
#----------------------- 1.Reading data & Processing it ----------------------#
#-----------------------------------------------------------------------------#

# The user is asked to enter the number of epochs, choice of activation function & number of mini-batches
epochs = int(input("Enter the number of epochs you want to run this for: "))
print("Choose your activation function for the hidden layer")
choice = int(input("Enter 1 for ReLU and 2 for Sigmoid: "))
batch_size=int(input("Enter Batch Size (1 or 2 or 3): "))

# Defining the path where the dataset resides
# Dr G PLEASE PROVIDE THE PATH FOR YOUR FILE HERE 
# path="C:/Users/17207/Desktop/"

# Defining the file name and reading the data
filename="TheData.csv"
df = pd.read_csv(filename)

# Splitting the data into Test & Train 
# We will randomly generate numbers and use these to index the rows in our dataframe into test and train
gen = np.random.rand(len(df)) < 0.8
DF = df[gen].reset_index(drop=True) # this is the training set
test = df[~gen].reset_index(drop=True)

# Removing our label and storing the rest as X in an array
X = np.array(DF.iloc[:, 1:])

# Assigning the label to y and transposing to make sure it is in the required shape
y = np.array(DF.iloc[:,0]).T
y = np.array([y]).T # this is to force y to retain its required shape

# Let us calculate the number of variables, number of categories and number of records
InputColumns = X.shape[1]               # Number of Variables ie., X1, X2, X3
NumberOfLevels = len(np.unique(y))     # Number of categories in the label ie., k
n = len(DF)                             # Number of rows in the data

# Setting Learning Rate
LR=.2 # Learning rate for weights
LRB=.01 # Learning rate for biases

# Creating one hot labels for y
temp = y
one_hot_labels = np.zeros((n, NumberOfLevels))
for i in range(n):
    one_hot_labels[i, temp[i]-1] = 1    
y = one_hot_labels



#-----------------------------------------------------------------------------#
#---------------------- 2. Neural Network Architecture -----------------------#
#-----------------------------------------------------------------------------#
class NeuralNetwork(object):
    def __init__(self):
        
        # Declaring variables to build the structure of the neural network
        self.InputNumColumns = InputColumns # Columns
        self.OutputSize = NumberOfLevels    # Categories
        self.HiddenUnits = 2                # one layer with h units
        self.n = n                          # number of training examples, n
        self.choice = choice                # user input on which activation function to use for the hidden layer
        
        # Initializing weights and biases for both layers
        ## print("Initialize NN\n")
        self.W1 = np.random.randn(self.InputNumColumns, self.HiddenUnits) # c by h  
        ## print("INIT W1 is\n", self.W1)
        self.W2 = np.random.randn(self.HiddenUnits, self.OutputSize) # h by o 
        ## print("W2 is:\n", self.W2)
        self.b = np.random.randn(1, self.HiddenUnits)
        ## print("The b's are:\n", self.b)        
        self.c = np.random.randn(1, self.OutputSize)
        ## print("The c is\n", self.c)

# Feedforward function will pass our X vector through the network once. This can be reused to get results for the test set.
    def FeedForward(self, X):
        self.z = (np.dot(X, self.W1)) + self.b 
        # X is n by c   W1  is c by h -->  n by h
        ## print("Z1 is:\n", self.z)
        
        self.h = self.activation_function(self.z) # activation function    shape: n by h
        ## print("H is:\n", self.h)
        
        self.z2 = (np.dot(self.h, self.W2)) + self.c # n by h  @  h by o  -->  n by o  
        ## print("Z2 is:\n", self.z2)
        
        # Using Softmax for the output activation
        output = self.Softmax(self.z2)  
        ## print("\nOutput Y^ is:\n", output)
        return output

# The function below takes in 'choice' from the user and applies the corresponding activation function
    def activation_function(self, s, deriv=False):
        if choice==1: # ReLU Activation
            s[s<0]=0.001
            return s
        if choice==2: # Sigmoid activation       
            if (deriv == True):
                return s * (1 - s)
            return 1/(1 + np.exp(-s))

# Activation function for the final/output layer
    def Softmax(self, M):
        ## print("M is\n", M)
        expM = np.exp(M)
        ## print("expM is\n", expM)
        SM=expM/np.sum(expM, axis=1)[:,None]
        ## print("SM is\n",SM )
        return SM 

# Backpropogation: This is where gradient descent recalculates weights and biases with 
# the objective of improving its next run such that the loss decreases
    def BackProp(self, X, y, output):
        self.LR = LR
        self.LRB= LRB  ## LR for biases

        # Y^ - Y
        self.output_error = output - y    

        # Leaving your note here as I found it very helpful Dr G :-)
        # NOTE TO READER........................
        # Here - we DO NOT multiply by derivative of Sig for y^ b/c we are using 
        # cross entropy and softmax for the loss and last activation
        # REMOVED # self.output_delta = self.output_error * self.Sigmoid(output, deriv=True) 
        # So the above line is commented out...............
        
        self.output_delta = self.output_error 
        # (Y^ - Y)(W2)
        self.D_Error_W2 = self.output_delta.dot(self.W2.T) #  D_Error times W2

        # (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        # We still use the Sigmoid on H
        self.H_D_Error_W2 = self.D_Error_W2 * self.activation_function(self.h, deriv=True) 
        # Note that * will multiply respective values together in each matrix
        
        #  XT  (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        self.X_H_D_Error_W2 = X.T.dot(self.H_D_Error_W2) # this is dW1
        
        # (H)T (Y^ - Y)
        self.h_output_delta = self.h.T.dot(self.output_delta) # this is for dW2
        
        ## print("the gradient :\n", self.X_H_D_Error_W2)
        ## print("the gradient average:\n", self.X_H_D_Error_W2/self.n)
        
        # Now that we have calculated our derivatives, let us update our weights & biases
        # Updating Weights in both layers
        self.W1 = self.W1 - self.LR*(self.X_H_D_Error_W2) # c by h  adjusting first set (input -> hidden) weights
        self.W2 = self.W2 - self.LR*(self.h_output_delta) 
        
        ## print("The sum of the b update is\n", np.sum(self.H_D_Error_W2, axis=0))
        ## print("The b biases before the update are:\n", self.b)
        # Updating Biases in both layers
        self.b = self.b  - self.LRB*np.sum(self.H_D_Error_W2, axis=0)
        ## print("The H_D_Error_W2 is...\n", self.H_D_Error_W2)
        ## print("Updated bs are:\n", self.b)
        self.c = self.c - self.LR*np.sum(self.output_delta, axis=0)
        ## print("Updated c's are:\n", self.c)
        
        ## print("The W1 is: \n", self.W1)
        ##print("The W1 gradient is: \n", self.X_H_D_Error_W2)
        ## print("The W1 gradient average is: \n", self.X_H_D_Error_W2/self.n)
        ## print("The W2 gradient  is: \n", self.h_output_delta)
        ## print("The W2 gradient average is: \n", self.h_output_delta/self.n)
        ## print("The biases b gradient is:\n",np.sum(self.H_D_Error_W2, axis=0 ))
        ## print("The bias c gradient is: \n", np.sum(self.output_delta, axis=0))
        
# This bit of code calls the feedforward and stores the output, then calls the backpropogation function        
    def TrainNetwork(self, X, y):
        output = self.FeedForward(X)
        self.BackProp(X, y, output)
        return output



#-----------------------------------------------------------------------------#
#---------------------- 3.Training & Testing our NN --------------------------#
#-----------------------------------------------------------------------------#
# Let us first instantiate our neural network
MyNN = NeuralNetwork()
TotalLoss=[] # Empty arrray for our total loss

# Batching: The following includes code that will split the training dataset into
# 2 or 3 parts depending on the user input. These parts will be sent to the feedforward and
# backpropogation functions individually and adjust weights and biases learnt from this
if batch_size>1:
    for i in range(epochs): 
##        loss_batch = 0              # total loss specifically for the current batch
        print("\n========================================================================")
        print("Epoch: ", i)
        for j in range(batch_size):
            # generates random numbers which will serve as indexes to subset our training X and Y arrays
            rand_index = np.random.randint(batch_size,size=n)   
            XB = X[rand_index==j]
            yB = y[rand_index==j]
            output=MyNN.TrainNetwork(XB, yB)
            ## print("The output is: \n", output)
            MaxValueIndex=np.argmax(output, axis=1)
            print('\nPrediction y^ on training data is', MaxValueIndex+1)
            # Using Categorical Cross Entropy...........
            loss = np.mean(-yB * np.log(output))  # We need y to place the "1" in the right place
##            loss_batch = loss_batch + loss
        print("\nThe total loss for this epoch is\n", loss)
        TotalLoss.append(loss)           
else:
    for i in range(epochs): 
        print("\n========================================================================")
        print("Epoch: ", i)
        # Calling our neural network to train on the mini-batches
        output=MyNN.TrainNetwork(X, y)
        ## print("The output is: \n", output)
        MaxValueIndex=np.argmax(output, axis=1)
        print('\nPrediction y^ on training data is', MaxValueIndex+1)
        # Using Categorical Cross Entropy...........
        loss = np.mean(-y * np.log(output))  # We need y to place the "1" in the right place
        print("\nThe total loss for this epoch is\n", loss)
        TotalLoss.append(loss)


    
#-----------------------------------------------------------------------------#
#------------------------- 4.Plotting our loss -------------------------------#
#-----------------------------------------------------------------------------#   
import matplotlib.pyplot as plt
fig1 = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, epochs)
ax.plot(x, TotalLoss)    
## print(y)



#-----------------------------------------------------------------------------#
#---------------- 5. Running Test data on our network ------------------------#
#-----------------------------------------------------------------------------# 
# Splitting our test dataset into X and Y to pass it to our prediction
X_test = np.array(test.iloc[:, 1:])
y_test = np.array(test.iloc[:,0]).T
y_actual = np.array([y_test]).T

# Passing X_test to our network with calculated weights and biases
y_pred = MyNN.FeedForward(X_test)
y_pred_ =np.argmax(y_pred, axis=1)
y_pred_ = y_pred_+1
print("\nOur predicted y^ on the test data is:\n")
print(y_pred_)

# Calculating the accuracy of our prediction
# We will subtract predicted y values from the actual y values and then sum up the 
# absolute values of the array which will give us the number of correct predictions
# Dividing this by the total size of y from the test data gives us accuracy
accuracy = np.sum(np.absolute(y_pred_-y_test))/len(y_test)*100
print("\nThe accuracy of our network on test data is: "+str(round(accuracy,2))+"%")


#-----------------------------------------------------------------------------#
#--------------------------6. Confusion Matrix -------------------------------#
#-----------------------------------------------------------------------------#        
# We will be using pandas to convert the arrays into pandas series. We do this because
# we can take advantage of the crosstab function pandas provides that neatly pivots
# all the unique values of the label into rows and columns (like a matrix) and counts the occurences
y_pred_ = pd.Series(y_pred_, name="Predicted")          # convert prediction of y into series
y_actual = pd.Series(y_actual.flatten(), name="Actual") # convert actual values of y into series
print("\nConfusion matrix for test data is:\n")
print(pd.crosstab(y_pred_,y_actual))                    # create a crosstab that pivots both series



#-----------------------------------------------------------------------------#
#--------------------------7. Results & Conclusion ---------------------------#
#-----------------------------------------------------------------------------# 

# The above program was run for 100 epochs for the following 6 combinations 
# The accuracy on the test data is as follows:
#                Accuracy for functions
#Batch	           RELU	           Sigmoid
#1	              45.65%	         43.55%
#2	              44.25%	         45.28%
#3	              42.63%	         53.11%

# It seems that the batch size increase does not bode well for ReLU, but for Sigmoid this is NOT TRUE
# One fact to keep in mind is that these numbers are highly volatile and further investigation
# needs to be done, with a larger number of epochs to minimize the fluctuations
