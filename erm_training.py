import cvxpy as cp
import numpy as np
from constants import *
from helpers import *

def train_erm(X, L_mat, U_mat=None, groups=None, lamb=0):
    """ In our current model, ERM does not depend on the mixing co-efficients
        So this is basically deterministic
    """
    L_X = np.matmul(X, L_mat.T)
    L_X = normalize(L_X, axis=1, norm='l1')
    
    n, d = L_X.shape
    n, m = X.shape

    # Constructing argmax/min labels used repeatedly
    y = np.argmin(L_X, axis = 1)

    learned_betas = []
    learned_predictions = []
    def_alphas = get_default_alpha_arr(K)

    # Run code to compute the mixture iteratively
    for k in range(K):
        Beta = cp.Variable((d,m))  # The parameters of the one-vs-all classifier
        
        # Solve relaxed convexified optimization problem
        loss_objective = 0
        for i in range(n):
            # construct list with entries L(x_i, y) + beta_y^T x_i - beta_{y_i}^T x_i; for each y and y_i defined appropriately
            loss_objective += get_convex_version(X, L_X, Beta, y, i)

        objective = cp.Minimize((1/n)*loss_objective)
        prob = cp.Problem(objective)
        
        # Solving the problem
        try:
            results = prob.solve(solver=cp.SCS, verbose=False)#, feastol=1e-5, abstol=1e-5)
        except:
            return 0, 0, 0, 0
        Beta_value = np.array(Beta.value)
        learned_betas.append(Beta_value)

        all_predictions = predictions(Beta_value, X)
        learned_predictions.append(all_predictions)

    learned_pred_group = None
    if groups is not None and U_mat is not None:
        num_groups = len(groups.keys())
        learned_pred_group = {h:[] for h in range(num_groups)} 
        for k in range(K):
            for h in range(num_groups):
                learned_pred_group[h].append(predictions(Beta_value, groups[h]))

    return learned_betas, learned_predictions, learned_pred_group, def_alphas

def test_erm():
    # d (classes) = 5, n(samples) = 3, m(feature_size) = 4
    alphas = get_default_alpha_arr(K)
    train_X = np.array([[0.4, 0.3, 0.5, 0.1], \
                        [0.3, 0.1, 0.7, 0.9], \
                        [1.1, 0.4, 0.5, 1.2]]) # 3 x 4

    L = np.array([[0.3, 0.1, 0.4, 0.5], [0.2, 0.4, 0.1, 0.5], [0.3, 0.55, 0.7, 0.8], \
            [0.1, 0.1, 0.5, 0.3], [0.3, 0.4, 0.5, 1.0]])
    learned_betas, learned_predictions, _, alphas = train_erm(train_X, L)
    final_loss = compute_final_loss(alphas, L, train_X, learned_predictions)
    opt_loss = get_optimal_loss(L, train_X)
    assert((final_loss - opt_loss) <= eps)
    print("TEST 1: PASSED")
    
    train_X = np.array([[0.4, 0.3, 0.5, 0.1], \
                        [0.3, 0.1, 0.7, 0.9], \
                        [1.1, 0.4, 0.5, 1.2],
                        [0.3, 0.5, 1.2, 1.3],
                        [1.3, 0.2, 0.7, 0.9],
                        [1.7, 1.5, 1.9, 0.3],
                        [2.2, 0.9, 1.7, 0.3],
                        [0.1, 0.2, 1.9, 1.3],
                        ]) # 5 x 4
    L = np.array([[0.3, 0.1, 0.4, 0.2],\
                  [0.2, 0.4, 0.1, 0.7],\
                  [0.3, 0.55, 0.7, 0.3],\
                  [0.4, 0.1, 0.4, 0.2]])
    learned_betas, learned_predictions, _, alphas = train_erm(train_X, L)
    final_loss = compute_final_loss(alphas, L, train_X, learned_predictions)
    opt_loss = get_optimal_loss(L, train_X)
    assert((final_loss - opt_loss) <= eps)
    print("TEST 2: PASSED")

def size_test():
    ns_ = 20
    print("\n\nRunning Size Test")

    train_X = generate_data(ns_, m, 'uniform')
    L = generate_loss_matrix(d, m, 'uniform')
    U = generate_utility_matrix(d, m, 'uniform')
    
    learned_betas, learned_predictions, _, alphas = train_erm(train_X, L)
    final_loss = compute_final_loss(alphas, L, train_X, learned_predictions)
    opt_loss = get_optimal_loss(L, train_X)
    
    print("Optimal loss is: ", opt_loss)
    print("ERM loss is: ", final_loss)

if __name__ == "__main__":
    test_erm()
    size_test()

