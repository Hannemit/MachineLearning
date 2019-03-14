import numpy as np 


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sigmoid_gradient(z):
    """
    Derivative of sigmoid(z)
    """
    sigmoid_grad = sigmoid(z)*(1 - sigmoid(z))
    return sigmoid_grad


def roll(nn_params, layer_sizes):
    """
    nn_params: long array of weights
    layer_sizes: vector of length 3 containing input, hidden, and output layer sizes (# nodes)

    returns:
        weight_mat_1, weight_mat_2, two weight matrices for going from layer 1 to layer 2, and layer 2 to layer 3.
    """
    n_in, n_hid, n_out = layer_sizes

    len_t1 = n_hid * (n_in + 1)
    len_t2 = n_out * (n_hid + 1)

    weight_mat_1 = np.reshape(nn_params[:len_t1], (n_hid, n_in + 1))
    weight_mat_2 = np.reshape(nn_params[len_t1:], (n_out, n_hid + 1))
    return weight_mat_1, weight_mat_2


def unroll(weights_mat_1, weight_mat_2):
    """
    Take in two weight matrices, output one long array of weights. Undo this again with 'Roll'
    """
    reshaped_weights_1 = np.reshape(weights_mat_1, np.size(weights_mat_1))
    reshaped_weights_2 = np.reshape(weight_mat_2, np.size(weight_mat_2))
    combined_weights = np.concatenate((reshaped_weights_1, reshaped_weights_2))
    nn_params = np.reshape(combined_weights, np.size(combined_weights))
    return nn_params


def get_activations(weights_1, weights_2, input_data):

    # Now calculate the activations
    a1 = input_data # activations of layer 1 are just the input values
    a1 = np.insert(a1, 0, 1, axis=1) # First, add col of ones to input_data, which are the bias units
    a1 = np.transpose(a1)
    
    a2 = sigmoid(np.dot(weights_1, a1))
    a2 = np.insert(a2, 0, 1, axis=0) # add row of ones
    a3 = sigmoid(np.dot(weights_2, a2)) # our outputs, a Kxm vector (K = nb classes)
    
    return [a1, a2, a3]


def one_hot_encode(y, num_classes, m):
    labels = np.zeros((num_classes, m))
    labels[y, range(m)] = 1 # y = 1,2, ... 10 for images 1,2, .. 0
    
    return labels


def get_cost(labels, prediction_mat, weight_mat_1, weight_mat_2, reg_param, num_train_points):
    """
    Calculate the cost of the current configuration.
    param labels: Kxm matrix with true labels. K = number of classes, num_train_points = number of training examples.
    each column contains all zeros and a single one at the correct class 
    param prediction_mat: Kxm matrix containing predictions, each column is single training example.

    param weight_mat_1: s1xs2 matrix, weight matrix connecting input layer to first hidden layer. s1 is number of
    nodes in hidden layer, s2 number of inputs + bias.
    param weight_mat_2: Kx(s1+1) matrix, weight matrix connecting hidden layer to output layer. K is number of output
    classes, s1 number of nodes in hidden layer.
    param reg_param: float, regularisation parameter
    param num_train_points: int, number of training examples

    returns: 
    J: float, cost of current configuration
    """
    # weight_mat_1, weight_mat_2 = roll(nn_params, layer_sizes)

    # Now calculate overall cost
    pred_mat = -labels*np.log(prediction_mat) - (1 - labels)*np.log(1.0 - prediction_mat)
    cost = 1.0/num_train_points*np.sum(pred_mat)

    # Add regularization term for cost. No regularizaton for the bias units (indexed by 0)
    # reg_theta1 = np.copy(weight_mat_1[:, 1:])
    # reg_theta2 = np.copy(weight_mat_2[:, 1:])
    # J_reg = reg_param/(2.0*num_train_points)*(np.sum(reg_theta1**2) + np.sum(reg_theta2**2))

    cost_regularize = reg_param/(2.0*num_train_points)*(np.sum(weight_mat_1[:, 1:]**2) + np.sum(weight_mat_2[:, 1:]**2))

    cost += cost_regularize
    return cost


def get_cost_gradient(nn_params, layer_sizes, input_data, y, reg_param):
    
    """
    layer_sizes: vector containing number of units in each of the layers, e.g. [400, 4, 10] (input, hidden, output)

    get_cost_gradient Implements the neural network cost function for a two layer
    neural network which performs classification
    """
    weight_mat_1, weight_mat_2 = roll(nn_params, layer_sizes)
    num_train_points = input_data.shape[0]
    # Setup some useful variables (num_train_points = number of training examples)

    # activations is vector [a1, a2, a3] where a3 are final outputs
    activations = get_activations(weight_mat_1, weight_mat_2, input_data)
    
    # One hot encode the correct labels
    labels = one_hot_encode(y, layer_sizes[-1], num_train_points)
    
    # Now calculate overall cost
    cost = get_cost(labels, activations[-1], weight_mat_1, weight_mat_2, reg_param, num_train_points)
    
    # Gradient using back propagation
    grad = perform_back_prop(labels, activations, weight_mat_1, weight_mat_2, reg_param, num_train_points)

    return cost, grad


def perform_back_prop(labels, activations, weight_mat_1, weight_mat_2, reg_param, num_train_points):
    """
    Function for backpropagation algorithm to provide the cost gradient. 

    returns:
        grad, long array containing gradients of cost with respect to each theta (Unroll used
                to create the long array).
    """
    # weight_mat_1, weight_mat_2 = roll(nn_params, layer_sizes)
    [a1, a2, a3] = activations
    delta3 = a3 - labels
    # delta3 is a Kxm matrix. One column denotes
    # one training example.  The length of the column is the number
    # of classes we have. So in a given column, delta3 gives the
    # %error of that output node, which is just the difference
    # between our prediction of that output node (a3) and
    # the actual value (labels).

    temp = np.dot(np.transpose(weight_mat_2), delta3)
    delta2 = temp[1:, :]*sigmoid_gradient(np.dot(weight_mat_1, a1))

    delta_1 = np.dot(delta2, np.transpose(a1))
    delta_2 = np.dot(delta3, np.transpose(a2))
    
    # Regularization terms. We want to add reg_param/num_train_points*Theta but
    # We do not want to regularize the thetas linked to the bias units
    # Which corresponds to the first column in theta.
    theta1_reg = np.copy(weight_mat_1)
    theta1_reg[:,0] = 0

    theta2_reg = np.copy(weight_mat_2)
    theta2_reg[:,0] = 0

    theta1_grad = 1.0/num_train_points*(delta_1 + reg_param*theta1_reg)
    theta2_grad = 1.0/num_train_points*(delta_2 + reg_param*theta2_reg)

    grad = unroll(theta1_grad, theta2_grad)
    # grad = [np.ravel(theta1_grad), np.ravel(theta2_grad)]
    return grad


def initialize_weights(n_in, n_out, eps):
    """
    Randomly initialise weights.
    """
    np.random.seed(3)
    w = np.random.uniform(size = (n_out, 1 + n_in))
    w = w*2*eps - eps
    return w


def predict(weight_mat_1, weight_mat_2, data_mat):
    """
    Predict the label of an input given a trained neural network
    param weight_mat_1: s1xs2 matrix, weight matrix connecting input layer to first hidden layer. s1 is number of
                    nodes in hidden layer, s2 number of inputs + bias.
    param weight_mat_2: Kx(s1+1) matrix, weight matrix connecting hidden layer to output layer. K is number of output
                    classes, s1 number of nodes in hidden layer.
    param data_mat: mxn matrix, each row is an image
    
    returns: 
        prediction_array: m-dim array containing predicted digit for each input sample
        prob: m-dim array, probability of an input belonging to its predicted class
    """
    outputs = get_activations(weight_mat_1, weight_mat_2, data_mat)[-1]

    m = data_mat.shape[0]
    # data_mat = np.insert(data_mat, 0, 1, axis=1) #add column of ones at start, bias units
    
    # h1 = sigmoid(np.dot(data_mat, weight_mat_1.T))
    # h1 = np.insert(h1, 0, 1, axis = 1)
    # h2 = sigmoid(np.dot(h1, weight_mat_2.T)) #each row is sample, each col is prob to be in certain class
    
    prediction_array = np.argmax(outputs, axis=0)  # indices of col with max value
    prob = outputs[prediction_array, np.arange(m)]  # probs of digit being the predicted label
    
    return prediction_array, prob
