import numpy as np
from scipy.stats import multivariate_normal

def InitialiseCentres(K, X):
	'''
	Initialise cluster centres
	param K: integer, number of clusters
	param X: dataset with each row a single example (mxn matrix)
	returns: 
		clus: Kxn matrix, each row is single centroid position
	'''
	m = len(X)
	rnd_idx = np.random.choice(m, size = K) #random indices
	clus = X[rnd_idx, :] #take data points as initial centroids	
	return clus


def Expectation(X, clus):
	'''
	Find closest cluster centroid to each data point
	param X: dataset with each row a single example (mxn matrix)
	param clus: cluster centroid coordinates, each row single cluster position (Kxn matrix)
	returns: 
		idx: array with cluster assignment index for each data point (m-dim)
		err: float giving total error (sum of squared distances) of current configuration
	'''
	K = len(clus)
	m = len(X) #number of training examples
	mat = np.zeros((m, K))
	for i in range(K):
		dist = np.sum((X - clus[i])**2, axis = 1)
		mat[:,i] = dist
	idx = np.argmin(mat, axis = 1) #closest cluster indices
	err = np.sum(mat[range(m), idx])

	return idx, err 

def FindCenters(X, idx, K):
	'''
	Find position of cluster centroids by computing means of the data points assigned to each centre.
	param X: dataset with each row a single example (mxn matrix)
	param idx: array with cluster assignment index for each data point (m-dim)
	param K: integer, number of clusters
	returns:
		clus: cluster centroid coordinates, each row single centroid position (Kxn matrix)   
	'''
	n = X.shape[1] #dimension of data points
	clus = np.zeros((K, n))
	for i in range(K):
		rows = X[idx == i]
		clus[i,:] = np.mean(rows, axis = 0)
	return clus 


def Responsibility(X, mu, cov, pi):
    '''
    Calculate responsibility matrix, r_ik, giving prob that datapoint i belongs to cluster k
    param X: (m x n) matrix, dataset with each row a single example
    param mu: (K x n) matrix, means of gaussian base dists, each row one dist
    param cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
    param pi: k-dim array, weights of each of the clusters    
    
    Returns r: (m x K) matrix, responsibility matrix
    '''
    m = X.shape[0]
    K = len(pi)
    r = np.zeros((m, K))

    for k in range(K):
        normal = multivariate_normal.pdf(X, mean = mu[k], cov = cov[k])     
        r[:, k] = pi[k]*normal #.pdf(X) takes pdf value per row
    r /= np.linalg.norm(r, axis=1, keepdims=True, ord = 1) #normalise r
    return r  

# def Responsibility(X, mu, cov, pi, nIter):
#     '''
#     Calculate responsibility matrix, r_ik, giving prob that datapoint i belongs to cluster k
#     param X: (m x n) matrix, dataset with each row a single example
#     param mu: (K x n) matrix, means of gaussian base dists, each row one dist
#     param cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
#     param pi: k-dim array, weights of each of the clusters    
    
#     Returns r: (m x K) matrix, responsibility matrix
#     '''
#     m = X.shape[0]
#     K = len(pi)
#     r = np.zeros((m, K))

#     #print(nIter)
#     for k in range(K):
#         #print(mu[k])
#         #print(cov[k])
        
#         normal = multivariate_normal.pdf(X, mean = mu[k], cov = cov[k])
        
#         #if nIter == 23:
#         #    assert False
#         #multivariate_normal.pdf(data,mean=params['mu0'], cov=params['sig0'])
#         #print(normal)
        
#         #we might be evaluating the gaussian at a point so far away we get underflow, prevent this.
#         # try:
#         #     pdf = normal.pdf(X)
#         # except FloatingPointError:
#         #     return -1
#         #pdf = normal.pdf(X)       
#         r[:, k] = pi[k]*normal #.pdf(X) takes pdf value per row
#     r /= np.linalg.norm(r, axis=1, keepdims=True, ord = 1) #normalise r
#     return r 


def Max(r, X):
    '''
    Finds the mu, cov and pi (weights) that maximise the data log likelihood
    param r: (m x K) matrix, responsibility matrix
    param X: (m x n) matrix, dataset with each row a single example
    
    returns: 
        pi:  k-dim array, weights of each of the clusters 
        mu:  (K x n) matrix, means of gaussian base dists, each row one dist
        cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
    '''
    m, K = r.shape
    n = X.shape[1]
    
    pi = np.mean(r, axis = 0)
    mu = np.zeros((K, n))
    cov = np.zeros((K, n, n))
    
    for k in range(K):
        r_k = np.sum(r[:,k]) #number
        mu[k, :] = np.sum(r[:, k].reshape(m,1)*X, axis = 0)/r_k 
        mu_dot =  np.dot(mu[k,:].reshape(n,1), mu[k,:].reshape(1,n)) #mu col vec times mu row vec , nxn
        
        temp = np.einsum('ab,ac->abc',X,X)
        temp2 = np.einsum('a,abc->bc',r[:,k],temp)
        cov[k] = temp2/r_k - mu_dot #nxn
    
    return pi, mu, cov


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    Source: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def Log_llh(r, X, mu, cov, pi ):
    '''
    Calculate the expected data log likelihood
    param r: (m x K) matrix, responsibility matrix
    param X: (m x n) matrix, dataset with each row a single example
    param mu: (K x n) matrix, means of gaussian base dists, each row one dist
    param cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
    param pi: k-dim array, weights of each of the clusters        

    returns:
        log_llh: float, expected complete data log likelihood
                 Eq. based on 'Machine learning: a probabilistic  perspective', Murphy
    '''
    llh_1 = np.sum(r*np.log(pi))
    llh_2 = 0
    K = len(pi)
    for k in range(K):
        normal = multivariate_normal.pdf(X, mean = mu[k], cov = cov[k])
        llh_2 += np.sum(r[:,k]*np.log(normal)) #.pdf(X) takes pdf value per row  

    log_llh = llh_1 + llh_2
    return log_llh

# def Log_llh(r, X, mu, cov, pi ):
#     '''
#     Calculate the expected data log likelihood
#     param r: (m x K) matrix, responsibility matrix
#     param X: (m x n) matrix, dataset with each row a single example
#     param mu: (K x n) matrix, means of gaussian base dists, each row one dist
#     param cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
#     param pi: k-dim array, weights of each of the clusters        

#     returns:
#         log_llh: float, expected complete data log likelihood
#                  Eq. based on 'Machine learning: a probabilistic  perspective', Murphy
#     '''
#     llh_1 = np.sum(r*np.log(pi))
#     llh_2 = 0
#     K = len(pi)
#     for k in range(K):
#         normal = multivariate_normal.pdf(X, mean = mu[k], cov = cov[k])

#         #we might be evaluating the gaussian at a point so far away we get underflow, prevent this.
#         # try:
#         #     pdf = normal.pdf(X)
#         # except FloatingPointError:
#         #     return -1
#         llh_2 += np.sum(r[:,k]*np.log(normal)) #.pdf(X) takes pdf value per row  
#     log_llh = llh_1 + llh_2
    
#     #log_llh = 3
#     return log_llh


def PlotContours(mu, cov, X, axs, cmap = None, alpha = None):
    '''
    Function to plot contours of a bivariate Gaussian.
    param mu: 2D array with coordinates of Gaussian mean
    param cov: 2x2 covariance matrix of Gaussian 
    param X: mx2 matrix, one row is single datapoint, used for plot limits
    param cmap: matplotlib colour map 
    param alpha: float between 0 and 1
    param axs: axes instance

    returns: a plot
    '''
    if cmap is None:
        cmap = 'winter'
    if alpha is None:
        alpha = 0.3

    xmin, ymin = np.min(X, axis = 0)
    xmax, ymax = np.max(X, axis = 0)

    #make grid to plot contours for multivariate Gaussian
    xs = np.arange(xmin-1, xmax+1, 0.1)
    ys = np.arange(ymin-1, ymax+1, 0.1)
    Xs, Ys = np.meshgrid(xs, ys)

    pos = np.empty(Xs.shape + (2,))
    pos[:, :, 0] = Xs
    pos[:, :, 1] = Ys   

    Z = multivariate_gaussian(pos, mu=mu, Sigma=cov)
    axs.contour(Xs, Ys, Z, cmap=cmap, alpha= alpha)

def Norm_data(X):
    '''
    Normalise data, subtract mean and divide by std
    '''
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i], ddof = 1)

    return X