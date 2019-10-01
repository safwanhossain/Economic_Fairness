import cvxpy as cp
import numpy as np
import time
from constants import *
from helpers import *
from erm_training import train_erm

def train_erm_welfare(X, L_mat, U_mat, groups=None, lamb=0.5):
    """ In our current model, ERM does not depend on the mixing co-efficients
        So this is basically deterministic
    """
    L_X = np.matmul(X, L_mat.T)
    U_X = np.matmul(X, U_mat.T)
    n, d = L_X.shape
    n, m = X.shape

    # Constructing argmax/min labels used repeatedly
    y = np.argmin(L_X, axis = 1)
    s = np.argmin(U_X, axis = 1)
    b = np.argmax(U_X, axis = 1)

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

        welfare_objective = 0
        for i in range(n):
            USFii = 0
            for t in range(k):
                USFii += def_alphas[t]*U_X[i, learned_predictions[t][i]]

            max_val = np.amax(U_X)
            conjugate_UX = max_val*np.ones((n, d)) - U_X
            welfare_objective += USFii + def_alphas[k]*get_convex_version(X, conjugate_UX, Beta, b, i)

        objective = cp.Minimize((1/n)*loss_objective + lamb*(1/n)*welfare_objective)
        prob = cp.Problem(objective)
        
        # Solving the problem
        results = prob.solve()
        Beta_value = np.array(Beta.value)
        learned_betas.append(Beta_value)

        all_predictions = predictions(Beta_value, X)
        learned_predictions.append(all_predictions)

    # Now Iteratively learn the lambda values
    alphas = cp.Variable(K)
    alpha_loss = 0
    alpha_losses = []
    constraints = []
    for i in range(K):
        for i in range(n):
            alpha_loss += L_X[i, learned_predictions[k][i]] - lamb*U_X[i, learned_predictions[k][i]]
        alpha_losses.append(alpha_loss)
    objective = cp.Minimize(cp.sum(cp.multiply(alpha_losses, alphas)))
    
    constraints.append(cp.sum(alphas) == 1)
    for k in range(K):
        constraints.append(alphas[k] >= 0)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    opt_alphas = np.array(alphas.value).flatten()

    learned_pred_group = None
    if groups is not None:
        num_groups = len(groups.keys())
        learned_pred_group = {h:[] for h in range(num_groups)} 
        for k in range(K):
            for h in range(num_groups):
                learned_pred_group[h].append(predictions(Beta_value, groups[h]))
    
    return learned_betas, learned_predictions, learned_pred_group, opt_alphas

def test_erm_welfare():
    print("Test 1: n=8, d=4, m=4, Loss mat, Utility is identity")
    train_X = np.array([[0.4, 0.3, 0.5, 0.1], \
                        [0.3, 0.1, 0.7, 0.9], \
                        [1.1, 0.4, 0.5, 1.2],
                        [0.3, 0.5, 1.2, 1.3],
                        [1.3, 0.2, 0.7, 0.9],
                        [1.7, 1.5, 1.9, 0.3],
                        [2.2, 0.9, 1.7, 0.3],
                        [0.1, 0.2, 1.9, 1.3],
                        ]) # 8 x 4
    L = np.array([[0.3, 0.1, 0.4, 0.2],\
                  [0.2, 0.4, 0.1, 0.7],\
                  [0.3, 0.55, 0.7, 0.3],\
                  [0.4, 0.1, 0.4, 0.2]])
    U = np.array([[0.4, 0.2, 0.3, 0.4],\
                  [0.5, 0.1, 0.9, 0.5],\
                  [0.3, 0.4, 0.6, 0.5],\
                  [0.2, 0.3, 0.6, 0.1]])
    U_X = np.matmul(train_X, U.T)
    n, d = U_X.shape
    max_val = np.amax(U_X)

    learned_betas, learned_predictions, _, opt_alphas = train_erm(train_X, L, U, groups=None, lamb=0)
    final_loss = compute_final_loss(opt_alphas, L, train_X, learned_predictions)
    welfare = compute_welfare(opt_alphas, U, train_X, learned_predictions)
    opt_loss = get_optimal_loss(L, train_X)

    print("Optimal loss is: ", opt_loss)
    print("ERM loss is: ", final_loss)
    print("Welfare is: ", welfare)
    
    #conjugate_UX = max_val*np.ones((n, d)) - U_X
    #learned_betas, learned_predictions, _, opt_alphas = train_erm(train_X, conjugate_UX)
    #best_welfare = compute_welfare(opt_alphas, U, train_X, learned_predictions)
    #print("Best possible welfare is: ", best_welfare)

    learned_betas, learned_predictions, _, opt_alphas = train_erm_welfare(train_X, L, U, lamb=10)
    final_loss = compute_final_loss(opt_alphas, L, train_X, learned_predictions)
    welfare = compute_welfare(opt_alphas, U, train_X, learned_predictions)
    print("ERM with welfre loss is: ", final_loss)
    print("Welfare is: ", welfare)
    #assert(best_welfare - welfare < 1e-3)
    #print("TEST PASSED")

def size_test():
    n = 200
    m = 16
    d = 5
    K = 4
    print("\n\nRunning Size Test")

    train_X = generate_data(n, m, 'uniform')
    L = generate_loss_matrix(d, m, 'uniform')
    U = generate_utility_matrix(d, m, 'uniform')
    U_X = np.matmul(train_X, U.T)
    
    max_val = np.amax(U_X)
    
    learned_betas, learned_predictions, _, alphas = train_erm(train_X, L)
    final_loss = compute_final_loss(alphas, L, train_X, learned_predictions)
    welfare = compute_welfare(alphas, U, train_X, learned_predictions)
    opt_loss = get_optimal_loss(L, train_X)
    
    print("Optimal loss is: ", opt_loss)
    print("ERM loss is: ", final_loss)
    print("Welfare is: ", welfare)
    
    #conjugate_UX = max_val*np.ones((n, d)) - U_X
    #learned_betas, learned_predictions, _, alphas = train_erm(train_X, conjugate_UX)
    #best_welfare = compute_welfare(alphas, U, train_X, learned_predictions)
    #print("Best possible welfare is: ", best_welfare)
    
    learned_betas, learned_predictions, _, opt_alphas = train_erm_welfare(train_X, L, U, lamb=10)
    final_loss = compute_final_loss(opt_alphas, L, train_X, learned_predictions)
    welfare = compute_welfare(opt_alphas, U, train_X, learned_predictions)
    print("ERM with welfre loss is: ", final_loss)
    print("Welfare is: ", welfare)
    #assert(best_welfare - welfare < 1e-3)
    #print("TEST PASSED")

if __name__ == "__main__":
    test_erm_welfare()
    size_test()
