import operator as op 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split

def ncr(n, r):
    '''
    n choose r function
    '''
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def costFunction(theta, X, y, lamb):
    '''
    Compute the cost and gradient for logistic regression.
    lamb: regularization parameter (float)
    y:    true labels (mx1 vector, m = number training examples)
    X:    input matrix, (mxn matrix with n number of features)
    theta: parameters (nx1 vector)
    '''
    
    m = len(y) #number of training examples
    
    #predictions given X and theta
    pred = sigmoid(np.dot(X, theta))
    
    #cost
    if 0 in pred or 1 in pred:
        print(theta)
        print(X)
        print(pred)
        assert False
     
    #Cost function, the `normal' part and the regularization part
    J_norm = -1.0/m*np.sum(y*np.log(pred) + (1 - y)*np.log(1 - pred))    
    J_reg = lamb/(2.0*m)*(np.sum(theta**2) - theta[0]**2)
    J = J_norm + J_reg
   
    #gradient of the cost function.
    grad_norm = 1.0/m*np.reshape(np.sum((pred - y)*X, axis = 0), (np.shape(X)[1], 1)) 
    grad_reg = float(lamb)/m*theta
    grad_reg[0] = 0
    
    grad = grad_norm + grad_reg 
    
    return J, grad


def mapFeatures(X, degree):
    '''
    Map to polynomial features. We have the input features 
    which are mapped to higher order features.
    
    This function returns a new feature array with more features, consisting of 
    x1, x2, ..xn, x1**2, x2**2, ..., xn**2, x1*x2, x1*x2**2", etc.. 
    '''
    
    nf = np.shape(X)[1] #number of features is number of cols
    nr = np.shape(X)[0] #number of rows
    total = ncr(nf+degree, degree) #number of new features
    out = np.ones((nr, total))

    for i2 in range(nr):
        l = 0
        for k in range(degree):
            for i in itertools.combinations_with_replacement(X[i2,:], k+1): 
                prod = 1
                for j in range(k+1):
                    prod = prod*i[j]
                out[i2,l+1] = prod
                l += 1
    return out, total


def GradientDescent(X, y, theta, lr, lamb, nIters):
    m = float(len(y)) #number of training examples
    costValues = np.zeros(nIters)
    grad = costFunction(theta, X, y, lamb)[1] #first gradient
    for i in range(nIters):       
        theta = theta - lr*grad #/m*np.reshape(np.sum((np.dot(X, theta) - y)*X, axis = 0), (np.shape(X)[1], 1))
        costValues[i], grad = costFunction(theta, X, y, lamb)

    return theta, costValues


def PlotData(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plt.plot(X[pos,1], X[pos,2], 'b+', linewidth =2)
    plt.plot(X[neg,1], X[neg,2], 'go', linewidth =2)


def plotDecisionBoundary(theta, X, y, degree):
    '''
    plot the decision boundary. This only works if there are two features.
    '''

    if np.shape(X)[1] <= 3: #if 2 or less features (first col of X is all ones)
        #Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:,1])*0.9,  max(X[:,1])*1.1]; #just two values of the first features

        #Calculate the decision boundary line. In this case we only have three thetas
        #Decision boundary is defined by theta[0] + theta[1]*x1 + theta[2]*x2 = 0. Here calculate x2
        plot_y = -1.0/theta[2]*(theta[1]*plot_x + theta[0]);

        #Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
        plt.xlim(plot_x)
        plt.ylim(min(X[:,2])*0.9, max(X[:,2])*1.1)
    else:
        #Here is the grid range
        u = np.linspace(-2, 2.5, 70);
        v = np.linspace(-2, 2.5, 70);
        z = np.zeros((len(u), len(v)));
        A = np.zeros((1,2))

        #Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                A[0,0] = u[i]
                A[0,1] = v[j]
                #print('A is {}').format(A)
                z[i,j] = np.dot(mapFeatures(A, degree)[0], theta)

        z = np.transpose(z) #important to transpose z before calling contour
        if np.isnan(np.sum(z)):
            print('isnan')
        # Plot z = 0
        plt.contour(u, v, z, [0], linewidth = 2)    
    
    
def predictOneVsAll(all_theta, X):
    '''
    PREDICT Predict the label for a trained one-vs-all classifier. The labels 
    are in the range 1..K, where K = np.size(all_theta). 
    This function will return a vector of predictions
    for each example in the matrix X. Note that X contains the examples in
    rows. all_theta is a matrix where the i-th row is a trained logistic
    regression theta vector for the i-th class. 
    The function returns p, a vector with values from 1..K 
    (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2 for 4 examples) 
    '''

    #predictions
    pred_all = sigmoid(np.dot(all_theta, np.transpose(X)));                       
                       
    #pred_all has num_labels rows and many columns (one for each test example). 
    #Find maximum value in each column, which then is the prediction of that particular
    #training example
    p = np.argmax(pred_all, axis=0)
    
    return p    
    
    
def FractionCorrect(p, y):
    '''
    Given an array of predictions p and the true labels y, this function calculates
    the fraction of predictions that were correct.
    '''
    m = len(y)

    y = np.reshape(y, m)
    p = np.reshape(p, m)
    
    frac = np.sum(y == p)/np.float(m)
    
    return frac
    
def PredictAccuracy(X, y, lamb, lr, nIters, nb_c):
    '''
    Predicts the accuracy on training and test sets after averaging over splitting the data set randomly
    into different test and training data.
    '''
    n = 20
    nbFeatures =  np.shape(X)[1]
    pcorrect_train = np.zeros(n)
    pcorrect_test = np.zeros(n)
    
    
    for j in range(n):
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=j)

        #initialize parameters
        initial_theta = np.zeros((nbFeatures, 1))
        allTheta = np.zeros((nb_c, nbFeatures)) #to store the learned theta parameters

        #for each class, find final theta values. We will use all-vs-one to predict the class
        for c in range(nb_c):
            finalTheta, costValues = GradientDescent(X_train, y_train==c, initial_theta, lr, lamb, nIters)
            allTheta[c,:] = np.reshape(finalTheta, nbFeatures)        

        #find out accuracy on training and test set
        p_train = predictOneVsAll(allTheta, X_train)
        pcorrect_train[j] = FractionCorrect(p_train, y_train)

        p_test = predictOneVsAll(allTheta, X_test)
        pcorrect_test[j] = FractionCorrect(p_test, y_test)
    
    return pcorrect_train, pcorrect_test
        
    
    