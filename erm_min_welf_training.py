import cvxpy as cp
import numpy as np
import time
from constants import *
from helpers import *
from erm_training import train_erm

def train_erm_min_welfare(X, L_mat, U_mat, groups, lamb=1000):
    L_X = np.matmul(X, L_mat.T)
    U_X = np.matmul(X, U_mat.T)
    n, d = L_X.shape
    n, m = X.shape
        
    # For each group, compute its U_X and L_X matrix
    num_groups = len(groups.keys())
    group_ids = groups.keys()
    group_sizes = [groups[i].shape[0] for i in range(num_groups)]
    group_LX, group_UX = {}, {}
    for i in range(num_groups):
        group_LX[i] = np.matmul(groups[i], L_mat.T)
        group_UX[i] = np.matmul(groups[i], U_mat.T)

    # Constructing argmax/min labels used repeatedly
    y = np.argmin(L_X, axis = 1)
    s, b = [], []
    for i in range(num_groups):
        s.append(np.argmin(group_UX[i], axis = 1))
        b.append(np.argmax(group_UX[i], axis = 1))

    learned_betas = []
    learned_predictions = {h:[] for h in range(num_groups)} 
    learned_predictions_all = []
    def_alphas = get_default_alpha_arr(K)

    def get_min_welf_estimate(given_Beta):
        for i in range(num_groups):
            # First compute the utility group i has for itself
            USFii = 0
            concave_p = 0
            min_welf = 100
            for t in range(k):
                for li in range(group_sizes[i]):
                    USFii += def_alphas[t]*group_UX[i][li, learned_predictions[i][t][li]]
                    
            for li in range(group_sizes[i]):
                concave_version_p = np.min(group_UX[i][li, :] - np.matmul(given_Beta,groups[i][li,:])) + \
                        np.matmul(given_Beta[b[i][li],:],groups[i][li,:])
                concave_p += def_alphas[k]*concave_version_p

            USFii = USFii * (1/group_sizes[i])
            concave_p = concave_p * (1/group_sizes[i])
            min_welf = min(min_welf, USFii+concave_p)
        return min_welf
    
    # Run code to compute the mixture iteratively
    for k in range(K):
        Beta = cp.Variable((d,m))  # The parameters of the one-vs-all classifier
        
        # Solve relaxed convexified optimization problem
        loss_objective = 0
        for i in range(n):
            # construct list with entries L(x_i, y) + beta_y^T x_i - beta_{y_i}^T x_i; for each y and y_i defined appropriately
            loss_objective += get_convex_version(X, L_X, Beta, y, i)
        loss_objective = (1/n)*loss_objective

        # Our Envy-Free Objective is over groups - so iterate over them
        for i in range(num_groups):
            # First compute the utility group i has for itself
            USFii = 0
            concave_p = 0
            min_welf = 100
            for t in range(k):
                for li in range(group_sizes[i]):
                    USFii += def_alphas[t]*group_UX[i][li, learned_predictions[i][t][li]]
                    
            for li in range(group_sizes[i]):
                concave_version_p = cp.min(group_UX[i][li, :] - cp.matmul(Beta,groups[i][li,:])) + \
                        cp.matmul(Beta[b[i][li],:],groups[i][li,:])
                concave_p += def_alphas[k]*concave_version_p

            USFii = USFii * (1/group_sizes[i])
            concave_p = concave_p * (1/group_sizes[i])
            min_welf = cp.minimum(min_welf, USFii+concave_p)
    
        #objective = cp.Maximize((1/100)*((-1/10)*loss_objective + lamb*min_welf))
        objective = cp.Maximize(lamb*min_welf)
        prob = cp.Problem(objective)

        # Solving the problem
        try:
            #results = prob.solve(solver=cp.SCS, verbose=False)#, feastol=1e-5, abstol=1e-5)
            resuls = prob.solve(verbose=False)
        except:
            return 0, 0, 0, 0
        Beta_value = np.array(Beta.value)
        #print("beta: ", Beta_value)
        learned_betas.append(Beta_value)

        min_welfare_estimate = get_min_welf_estimate(Beta_value)
        #print("Min welfare estimate: " , min_welfare_estimate)
    
        all_predictions = predictions(Beta_value, X)
        learned_predictions_all.append(all_predictions)

        for h in range(num_groups):
            learned_predictions[h].append(predictions(Beta_value, groups[h]))


    # We now solve for the optimal alpha values
    alphas = cp.Variable(K)
    alpha_loss = 0
    alpha_losses = []
    min_welfares = []
    for k in range(K):
        alpha_loss = 0
        for i in range(n):
            alpha_loss += L_X[i, learned_predictions_all[k][i]]
        alpha_losses.append(alpha_loss)
        #print("Hello: ", alpha_losses) 
        for i in range(num_groups):
            utls = []
            total_ii_utl = 0
            for li in range(group_sizes[i]):
                total_ii_utl += group_UX[i][li, learned_predictions[i][k][li]]   
            #print("k: ", k, "i: ", i, "utl: ", total_ii_utl)
            utls.append(total_ii_utl/group_sizes[i])
        min_welfares.append(min(utls))

    objective = cp.Maximize(-cp.sum(cp.multiply(alpha_losses, alphas))\
            + lamb*cp.sum(cp.multiply(min_welfares, alphas)))
    
    constraints = []
    for k in range(K):
        constraints.append(alphas[k] >= 0)
    constraints.append(cp.sum(alphas) == 1)

    prob = cp.Problem(objective, constraints)
    #try:
    results = prob.solve(cp.SCS, verbose=False)#, feastol=1e-5, abstol=1e-5)
    #except:
    #return 0,0,0,0 
    opt_alphas = np.array(alphas.value).flatten()
    #print("XIS")
    #print(np.array(xis.value))
    return learned_betas, learned_predictions_all, learned_predictions, opt_alphas
      
def test_erm_equi():
    Lambda = 1000
    print("Group Envy Free Test: ")
    print("First compute ERM solution: ")
    train_X = np.array([[0.8, 0.3, 1.5, 0.1], \
                        [0.3, 1.1, 1.7, 0.9], \
                        [1.0, 1.4, 0.5, 1.2],
                        [0.3, 0.5, 1.2, 1.3],
                        [1.0, 0.2, 0.7, 0.9],
                        [0.7, 1.5, 1.9, 0.3],
                        [0.2, 0.9, 1.7, 0.3],
                        [0.1, 0.2, 1.9, 1.3],
                        [0.7, 0.277, 0.9, 1.1],
                        [1.0, 1.2, 0.7, 0.9],
                        [0.1, 0.8, 0.3, 0.5], \
                        ]) # 11 x 5
    group_dist = [0.25, 0.25, 0.25, 0.25]
    samples_by_group = define_groups(train_X, group_dist)
    L = np.array([[0.3, 0.1, 0.4, 0.2],\
                  [0.7, 0.4, 0.1, 0.7],\
                  [0.3, 0.55, 0.7, 0.3],\
                  [0.4, 0.1, 0.4, 0.2]])
    U = np.array([[.1, 0.3, 0.3, 0.9],\
                  [0.5, 0.9, 0.1, 0.5],\
                  [0.3, 0.55, 0.7, 0.3],\
                  [0.1, 0.9, 0.9, 0.1]])

    learned_betas, learned_pred_all, learned_pred_group, opt_alphas = \
            train_erm(train_X, L, U, samples_by_group)
    min_welfare = min_group_welfare(opt_alphas, U, samples_by_group, learned_pred_group) 
    loss = compute_final_loss(opt_alphas, L, train_X, learned_pred_all)
    optimal_loss = get_optimal_loss(L, train_X)
    
    print("Optimal Loss is:", optimal_loss)
    print("ERM loss is: ", loss)
    print("ERM min welfare: ", min_welfare)
    print("")
    
    print("Now do with min welfare constraint with lambda: ", Lambda)
    learned_betas, learned_pred_all, learned_pred_group, opt_alphas = \
            train_erm_min_welfare(train_X, L, U, samples_by_group, lamb=Lambda)
    cons_min_welfare = min_group_welfare(opt_alphas, U, samples_by_group, \
            learned_pred_group) 
    loss = compute_final_loss(opt_alphas, L, train_X, learned_pred_all)
    print("ERM-Min Welfare Loss is : ", loss)
    print("ERM-Min Welfare min welfare: ", cons_min_welfare)

def size_test():
    n = 60
    m = 16
    d = 5
    K = 4
    Lambda = 1000
    print("\nRunning Size Test")

    train_X = generate_data(n, m, 'uniform')
    group_dist = [0.25, 0.25, 0.25, 0.25]
    samples_by_group = define_groups(train_X, group_dist)
    L = generate_loss_matrix(d, m, 'uniform')
    U = generate_utility_matrix(d, m, 'uniform')
    
    erm_betas, learned_predictions, learned_pred_group, alphas = train_erm(train_X, L, U, \
            samples_by_group)
    final_loss = compute_final_loss(alphas, L, train_X, learned_predictions)
    min_welfare = min_group_welfare(alphas, U, samples_by_group, learned_pred_group) 
    opt_loss = get_optimal_loss(L, train_X)
    
    print("Optimal loss is: ", opt_loss)
    print("ERM loss is: ", final_loss)
    print("ERM min_welfare: ", min_welfare)
   
    start_time = time.time()
    cons_betas, learned_predictions, learned_predictions_group, opt_alphas = \
            train_erm_min_welfare(train_X, L, U, samples_by_group, lamb=Lambda)
    final_loss = compute_final_loss(opt_alphas, L, train_X, learned_predictions)
    min_welfare = min_group_welfare(opt_alphas, U, samples_by_group, learned_predictions_group) 
    print("ERM-min_welfare loss is: ", final_loss)
    print("ERM-min_welfare min welfare is: ", min_welfare)
    end_time = time.time()
    print("Time is: ", end_time - start_time)
    
    test_X = generate_data(100, m, 'uniform')
    group_dist = [0.25, 0.25, 0.25, 0.25]
    test_s_by_group = define_groups(test_X, group_dist)
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(erm_betas, test_X, test_s_by_group, K)
    min_welfare = min_group_welfare(opt_alphas, U, test_s_by_group, \
            st_learned_pred_group)
    print("ERM get this much min welfare on test: ", min_welfare)
    
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(cons_betas, test_X, test_s_by_group, K)
    min_welfare = min_group_welfare(opt_alphas, U, test_s_by_group, \
            st_learned_pred_group)
    print("ERM-min welfare get this much min welfare on test: ", min_welfare)

if __name__ == "__main__":
    test_erm_equi()
    size_test()
    #while True:
    #    size_test()

