import numpy as np
from scipy.stats import multivariate_normal


def initialise_centers(num_clusters, data_set):
    """

    :param num_clusters: integer, number of clusters
    :param data_set: dataset with each row a single example (mxn matrix)
    :return: clus, a Kxn matrix, each row is a single centroid position
    """
    m = len(data_set)
    rnd_idx = np.random.choice(m, size=num_clusters)  # random indices
    clus = data_set[rnd_idx, :]  # take data points as initial centroids
    return clus


def get_expectation(data_set, clus):
    """
    Find closest cluster centroid to each data point
    :param data_set: mxn numpy matrix, dataset with each row a single example
    :param clus: Kxn numpy matrix, cluster centroid coordinates with each row a single cluster position
    :return: idx, m-dim numpy array with cluster assignment index for each data point
             err, float giving total error of current configuration
    """
    num_clusters = len(clus)
    num_train_examples = len(data_set)

    mat = np.zeros((num_train_examples, num_clusters))
    for i in range(num_clusters):
        sum_sq_distance = np.sum((data_set - clus[i])**2, axis=1)
        mat[:, i] = sum_sq_distance
    idx_closest_cluster = np.argmin(mat, axis=1)
    err = np.sum(mat[range(num_train_examples), idx_closest_cluster])

    return idx_closest_cluster, err


def find_centers(data_set, idx, num_clusters):
    """
    Find position of cluster centroids by computing means of the data points assigned to each centre.
    param data_set: dataset with each row a single example (mxn matrix)
    param idx: array with cluster assignment index for each data point (m-dim)
    param num_clusters: integer, number of clusters
    returns:
        clus: 2d numpy array, cluster centroid coordinates, each row single centroid position (Kxn matrix)
    """
    n = data_set.shape[1]
    clus = np.zeros((num_clusters, n))
    for i in range(num_clusters):
        rows = data_set[idx == i]
        clus[i, :] = np.mean(rows, axis=0)
    return clus


def get_responsibility(data_set, mu, cov, pi):
    """
    Calculate responsibility matrix, r_ik, giving prob that datapoint i belongs to cluster k
    param data_set: (m x n) matrix, dataset with each row a single example
    param mu: (K x n) matrix, means of gaussian base dists, each row one dist
    param cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
    param pi: k-dim array, weights of each of the clusters    
    
    Returns r: (m x K) matrix, responsibility matrix
    """
    m = data_set.shape[0]
    num_classes = len(pi)
    r = np.zeros((m, num_classes))

    for k in range(num_classes):
        normal = multivariate_normal.pdf(data_set, mean=mu[k], cov=cov[k])
        r[:, k] = pi[k]*normal  # .pdf(data_set) takes pdf value per row
    r /= np.linalg.norm(r, axis=1, keepdims=True, ord=1)  # normalise r
    return r  

# def get_responsibility(X, mu, cov, pi, nIter):
#     """    Calculate responsibility matrix, r_ik, giving prob that datapoint i belongs to cluster k
#     param X: (m x n) matrix, dataset with each row a single example
#     param mu: (K x n) matrix, means of gaussian base dists, each row one dist
#     param cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
#     param pi: k-dim array, weights of each of the clusters    
    
#     Returns r: (m x K) matrix, responsibility matrix
#     """    m = X.shape[0]
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


def do_maximisation_step(resp_mat, data_set):
    """
    Finds the mu, cov and pi (weights) that maximise the data log likelihood
    param resp_mat: (m x K) matrix, responsibility matrix (K = num_classes)
    param data_set: (m x n) matrix, dataset with each row a single example
    
    returns: 
        pi:  k-dim array, weights of each of the clusters 
        mu:  (K x n) matrix, means of gaussian base dists, each row one dist
        cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
    """
    m, num_classes = resp_mat.shape
    n = data_set.shape[1]
    
    pi = np.mean(resp_mat, axis=0)
    mu = np.zeros((num_classes, n))
    cov = np.zeros((num_classes, n, n))
    
    for k in range(num_classes):
        r_k = np.sum(resp_mat[:, k])  # r_k is a scalar
        mu[k, :] = np.sum(resp_mat[:, k].reshape(m, 1)*data_set, axis=0)/r_k
        mu_dot = np.dot(mu[k, :].reshape(n, 1), mu[k, :].reshape(1, n))  # mu col vec times mu row vec , nxn
        
        temp = np.einsum('ab,ac->abc', data_set, data_set)
        temp2 = np.einsum('a,abc->bc', resp_mat[:, k], temp)
        cov[k] = temp2/r_k - mu_dot  # cov is nxn matrix
    
    return pi, mu, cov


def multivariate_gaussian(pos, mu, sigma):
    """
    Calculate the multivariate Gaussian distribution on array pos.
    Source: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    :param pos: numpy array, constructed by packing the meshed arrays of variables
                x1, x2, .. xk into its last dimension
    :param mu: numpy array, means of gaussians
    :param sigma: multidimensional numpy array, covariance matrix of gaussians
    :return: multivariate gaussian distribution on the array 'pos;
    """
    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    n = np.sqrt((2*np.pi)**n * sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, pos-mu)

    return np.exp(-fac / 2.0) / n


def get_log_llh(resp_mat, data_set, mu, cov, pi):
    """
    Calculate the expected data log likelihood
    param resp_mat: (m x K) matrix, responsibility matrix
    param data_set: (m x n) matrix, dataset with each row a single example
    param mu: (K x n) matrix, means of gaussian base dists, each row one dist
    param cov: (K x n x n) matrix, cov[k] is cov matrix of cluster k 
    param pi: k-dim array, weights of each of the clusters        

    returns:
        log_llh: float, expected complete data log likelihood
                 Eq. based on 'Machine learning: a probabilistic  perspective', Murphy
    """
    llh_1 = np.sum(resp_mat*np.log(pi))
    llh_2 = 0
    num_classes = len(pi)
    for k in range(num_classes):
        normal_pdf = multivariate_normal.pdf(data_set, mean=mu[k], cov=cov[k])
        llh_2 += np.sum(resp_mat[:, k]*np.log(normal_pdf))  # .pdf(data_set) takes pdf value per row

    log_llh = llh_1 + llh_2
    return log_llh


def plot_2d_gaussian_contours(mu, cov, data, axs, cmap='winter', alpha=0.3):
    """
    Function to plot contours of a bivariate Gaussian.
    param mu: 2D numpy array with coordinates of Gaussian mean
    param cov: 2x2 numpy array, covariance matrix of Gaussian
    param data: mx2 numpy array, one row is single datapoint, used for plot limits
    param cmap: matplotlib colour map 
    param alpha: float between 0 and 1
    param axs: axes instance

    returns: a plot
    """

    xmin, ymin = np.min(data, axis=0)
    xmax, ymax = np.max(data, axis=0)

    # make grid to plot contours for multivariate Gaussian
    xs = np.arange(xmin-1, xmax+1, 0.1)
    ys = np.arange(ymin-1, ymax+1, 0.1)
    xs, ys = np.meshgrid(xs, ys)

    pos = np.empty(xs.shape + (2,))
    pos[:, :, 0] = xs
    pos[:, :, 1] = ys

    z = multivariate_gaussian(pos, mu=mu, sigma=cov)
    axs.contour(xs, ys, z, cmap=cmap, alpha=alpha)


def normalise_data(data):
    """
    Normalise data, subtract mean and divide by std
    :param data: 2D numpy array
    :return: normalised data
    """
    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - np.mean(data[:, i]))/np.std(data[:, i], ddof=1)
    return data
