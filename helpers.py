import numpy as np
import cvxpy as cp
from sys import argv
import os
from scipy import stats
from scipy.special import softmax

# generate a numpy matrix of size \R num_samples * m, dist can be "uniform" or "normal"
def generate_data(n_s, m, dist):
    if (dist == 'uniform'):
        X = np.random.uniform(low=0.0, high=1.0, size=(n_s, m))
    elif(dist == 'normal'):
        X = stats.truncnorm.rvs(-1.0, 1.0, 0.0, 1.0, size=(n_s, m))
    return X

# generate matrix L of size d x m according to distribution dist
def generate_loss_matrix(d, m, dist):
    if (dist == 'uniform'):
        L = np.random.uniform(low=0.0, high=1.0, size=(d, m))
    elif(dist == 'normal'):
        L = stats.truncnorm.rvs(-1.0, 1.0, 0.0, 1.0, size=(d, m))
    return L

# generate matrix U of size d x m according to distribution dist
def generate_utility_matrix(d, m, dist):
    if (dist == 'uniform'):
        U = np.random.uniform(low=0.0, high=1.0, size=(d, m))
    elif (dist == 'normal'):
        U = stats.truncnorm.rvs(-1.0, 1.0, 0.0, 1.0, size=(d, m))
    return U

def define_groups(X, group_dist):
    """ Given a vector where each element represents the portion of the population that should
        belong to each group, and the size of the vector represents the number of groups.

        Return a list of matricies, where each matrix contains all the individuals that belong
        to that group 
    """
    n, m = X.shape
    assert(sum(group_dist)-1 <= 1e-5)
    assert(min(group_dist) >= 0)

    low = 0
    #unif = np.random.uniform(size=(n,1)).reshape(n,)
    unif = np.linspace(0,1,n)
    samples_by_group = {}
    for group_id, val in enumerate(group_dist):
        high = low + val
        low_indicies = np.where(unif >= low)
        high_indicies = np.where(unif <= high)
        indicies = np.intersect1d(low_indicies, high_indicies)
        samples_by_group[group_id] = X[indicies,:]
        low = high

    return samples_by_group

def get_default_alpha_arr(K):
    alpha = []
    alpha_value = 1.0
    for k in range(K-1):
        alpha_value = alpha_value / 2
        alpha.append(alpha_value)
    alpha.append(alpha_value)
    return alpha

def get_optimal_loss(L_mat, X):
    loss_matrix = np.matmul(X, L_mat.T)
    (n, d) = loss_matrix.shape
    min_val = loss_matrix.min(axis=1)
    assert(min_val.shape == (n,))
    return sum(min_val)/n
    
def predictions(beta_value, all_X):
    return np.argmax(np.matmul(beta_value, all_X.T), axis = 0)

def get_all_predictions(beta_values, all_X, groups, K_):
    learned_predictions = []
    for k in range(K_):
        all_prediction = np.argmax(np.matmul(beta_values[k], all_X.T), axis = 0)
        learned_predictions.append(all_prediction)

    num_groups = len(groups.keys())
    learned_pred_group = {h:[] for h in range(num_groups)} 
    for k in range(K_):
        for h in range(num_groups):
            learned_pred_group[h].append(predictions(beta_values[k], groups[h]))
    
    return learned_predictions, learned_pred_group

def compute_final_loss(alphas, L_mat, X, learned_predictions):
    L_X = np.matmul(X, L_mat.T)
    final_loss = 0
    K = len(alphas)
    n = learned_predictions[0].shape[0]
    
    for k in range(K):
        for i in range(n):
            final_loss += alphas[k]*L_X[i, learned_predictions[k][i]]
    return final_loss/n

def compute_welfare(alphas, U_mat, X, learned_predictions):
    return compute_final_loss(alphas, U_mat, X, learned_predictions)

def group_utility(alphas, U_mat, group_i, group_j, pred_i, pred_j, same=False):
    ## Compute the utility group i has for group j's clasification
    UX_i = np.matmul(group_i, U_mat.T)
    UX_j = np.matmul(group_j, U_mat.T)
    ni = group_i.shape[0]
    nj = group_j.shape[0]

    if same:
        welf = compute_welfare(alphas, U_mat, group_i, pred_i)
        return welf
    else:
        welf = 0
        for t, alpha in enumerate(alphas):
            for li in range(ni):
                for lj in range(nj):
                    welf += alpha*UX_i[li, pred_j[t][lj]]
        return welf/(ni*nj) 

def total_group_envy(alphas, U_mat, groups, group_pred):
    violations = 0
    total_envy = 0
    num_groups = len(groups.keys())

    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                u_ii = group_utility(alphas, U_mat, groups[i], \
                        groups[i], group_pred[i], group_pred[i], same=True)
                u_ij = group_utility(alphas, U_mat, groups[i], \
                        groups[j], group_pred[i], group_pred[j])    
                total_envy += max(u_ij - u_ii, 0)
                if u_ij > u_ii + 1e-2:
                    violations += 1

    return total_envy, violations

def total_group_equi(alphas, U_mat, groups, group_pred):
    violations = 0
    total_equi = 0
    num_groups = len(groups.keys())

    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                u_ii = group_utility(alphas, U_mat, groups[i], \
                        groups[i], group_pred[i], group_pred[i], same=True)
                u_jj = group_utility(alphas, U_mat, groups[j], \
                        groups[j], group_pred[j], group_pred[j], same=True)    
                print("i: ", i, "j: ", j, "uii:", u_ii, "ujj: ", u_jj) 
                total_equi += abs(u_jj - u_ii)
                if total_equi >= 1e-3:
                    violations += 1

    return total_equi, violations

def get_convex_version(X, Mat_X, Beta, y, i):  
    return cp.max(Mat_X[i, :] + cp.matmul(Beta,X[i,:])) - cp.matmul(Beta[y[i],:], X[i,:])
