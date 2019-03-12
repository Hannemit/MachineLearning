import numpy as np 


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def unroll(weights_mat_1, weight_mat_2):
    """
    Take in two weight matrices, output one long array of weights. Undo this again with 'Roll'
    """
    reshaped_weights_1 = np.reshape(weights_mat_1, np.size(weights_mat_1))
    reshaped_weights_2 = np.reshape(weight_mat_2, np.size(weight_mat_2))
    combined_weights = np.concatenate((reshaped_weights_1, reshaped_weights_2))
    nn_params = np.reshape(combined_weights, np.size(combined_weights))
    return nn_params


def roll(nn_params, layer_sizes):
    """
    nn_params: long array of weights
    layer_sizes: vector of length 3 containing input, hidden, and output layer sizes (# nodes)

    returns:
        Theta1, Theta2, two weight matrices for going from layer 1 to layer 2, and layer 2 to layer 3.
    """
    n_in, n_hid, n_out = layer_sizes

    len_t1 = n_hid*(n_in + 1)
    len_t2 = n_out*(n_hid + 1)
    
    Theta1 = np.reshape(nn_params[:len_t1], (n_hid, n_in + 1))
    Theta2 = np.reshape(nn_params[len_t1:], (n_out, n_hid + 1))    
    return Theta1, Theta2


def sigmoid_gradient(z):
    """
    Derivative of sigmoid(z)
    """
    g = sigmoid(z)*(1 - sigmoid(z))
    return g


def get_activations(weights_1, weights_2, input_data):

    #Now calculate the activations
    a1 = input_data; #activations of layer 1 are just the input values
    a1 = np.insert(a1, 0, 1, axis=1) #First, add col of ones to input_data, which are the bias units
    a1 = np.transpose(a1)
    
    a2 = sigmoid(np.dot(weights_1, a1))
    a2 = np.insert(a2, 0, 1, axis=0) #add row of ones
    a3 = sigmoid(np.dot(weights_2, a2)) #our outputs, a Kxm vector (K = nb classes)
    
    return [a1, a2, a3]


def one_hot_encode(y, nb_class, m):
    labels = np.zeros((nb_class, m))
    labels[y, range(m)] = 1 #y = 1,2, ... 10 for images 1,2, .. 0
    
    return labels


def get_cost(labels, pred, Theta1, Theta2, lamb, m):
    """
    Calculate the cost of the current configuration.
    param labels: Kxm matrix with true labels. K = number of classes, m = number of training examples. 
    each column contains all zeros and a single one at the correct class 
    param pred: Kxm matrix containing predictions, each column is single training example.  

    param Theta1: s1xs2 matrix, weight matrix connecting input layer to first hidden layer. s1 is number of 
    nodes in hidden layer, s2 number of inputs + bias.
    param Theta2: Kx(s1+1) matrix, weight matrix connecting hidden layer to output layer. K is number of output
    classes, s1 number of nodes in hidden layer.
    param lamb: float, regularisation parameter
    param m: int, number of training examples

    returns: 
    J: float, cost of current configuration
    """
    #Theta1, Theta2 = roll(nn_params, layer_sizes)

    #Now calculate overall cost
    pred_mat = -labels*np.log(pred) - (1 - labels)*np.log(1.0 - pred)
    J = 1.0/m*np.sum(pred_mat)

    #Add regularization term for cost. No regularizaton for the bias units (indexed by 0) 
    #reg_theta1 = np.copy(Theta1[:, 1:])
    #reg_theta2 = np.copy(Theta2[:, 1:])
    #J_reg = lamb/(2.0*m)*(np.sum(reg_theta1**2) + np.sum(reg_theta2**2))

    J_reg = lamb/(2.0*m)*(np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))

    J += J_reg
    return J


def get_cost_gradient(nn_params, layer_sizes, X, y, lamb):
    
    """
    layer_sizes: vector containing number of units in each of the layers, e.g. [400, 4, 10] (input, hidden, output)

    get_cost_gradient Implements the neural network cost function for a two layer
    neural network which performs classification
    """
    Theta1, Theta2 = roll(nn_params, layer_sizes)
    m = X.shape[0]
    #Setup some useful variables (m = number of training examples)

    #activations is vector [a1, a2, a3] where a3 are final outputs
    activations = get_activations(Theta1, Theta2, X)
    
    #One hot encode the correct labels
    labels = one_hot_encode(y, layer_sizes[-1], m)
    
    #Now calculate overall cost
    J = Cost(labels, activations[-1], Theta1, Theta2, lamb, m)
    
    #Gradient using back propagation
    grad = perform_back_prop(labels, activations, Theta1, Theta2, lamb, m)

    
    return J, grad


def perform_back_prop(labels, activations, Theta1, Theta2, lamb, m):
    """
    Function for backpropagation algorithm to provide the cost gradient. 

    returns:
        grad, long array containing gradients of cost with respect to each theta (Unroll used
                to create the long array).
    """
    #Theta1, Theta2 = roll(nn_params, layer_sizes)
    [a1, a2, a3] = activations
    delta3 = a3 - labels
    #delta3 is a Kxm matrix. One column denotes
    #one training example.  The length of the column is the number
    #of classes we have. So in a given column, delta3 gives the 
    #%error of that output node, which is just the difference 
    #between our prediction of that output node (a3) and
    #the actual value (labels).

    temp = np.dot(np.transpose(Theta2), delta3)
    delta2 = temp[1:, :]*sigmoid_gradient(np.dot(Theta1, a1))

    Delta_1 = np.dot(delta2, np.transpose(a1))
    Delta_2 = np.dot(delta3, np.transpose(a2))
    
    #Regularization terms. We want to add lamb/m*Theta but 
    #We do not want to regularize the thetas linked to the bias units
    #Which corresponds to the first column in theta.
    theta1_reg = np.copy(Theta1) 
    theta1_reg[:,0] = 0

    theta2_reg = np.copy(Theta2)
    theta2_reg[:,0] = 0

    Theta1_grad = 1.0/m*(Delta_1 + lamb*theta1_reg);
    Theta2_grad = 1.0/m*(Delta_2 + lamb*theta2_reg);

    grad = unroll(Theta1_grad, Theta2_grad)
    #grad = [np.ravel(Theta1_grad), np.ravel(Theta2_grad)]    
    return grad


def initialize_weights(n_in, n_out, eps):
    """
    Randomly initialise weights.
    """
    np.random.seed(3)
    w = np.random.uniform(size = (n_out, 1 + n_in))
    w = w*2*eps - eps
    return w


def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    param Theta1: s1xs2 matrix, weight matrix connecting input layer to first hidden layer. s1 is number of 
                    nodes in hidden layer, s2 number of inputs + bias.
    param Theta2: Kx(s1+1) matrix, weight matrix connecting hidden layer to output layer. K is number of output
                    classes, s1 number of nodes in hidden layer.
    param X: mxn matrix, each row is an image
    
    returns: 
        pred: m-dim array containing predicted digit for each input sample
        prob: m-dim array, probability of an input belonging to its predicted class
    """
    outputs = get_activations(Theta1, Theta2, X)[-1]

    m = X.shape[0]
    #X = np.insert(X, 0, 1, axis=1) #add column of ones at start, bias units
    
    #h1 = sigmoid(np.dot(X, Theta1.T))
    #h1 = np.insert(h1, 0, 1, axis = 1)
    #h2 = sigmoid(np.dot(h1, Theta2.T)) #each row is sample, each col is prob to be in certain class
    
    pred = np.argmax(outputs, axis = 0) #indices of col with max value
    prob = outputs[pred, np.arange(m)] #probs of digit being the predicted label
    
    return pred, prob
