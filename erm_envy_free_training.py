import cvxpy as cp
import numpy as np
import time
from constants import *
from helpers import *
from erm_training import train_erm
from sklearn.preprocessing import normalize

"""
def procaccia_ef(X, L_mat, U_mat, groups, lamb=0.5):
    L = np.matmul(X, L_mat.T)
    U = np.matmul(X, U_mat.T)
    n, d = L.shape
    n, m = X.shape

    y = np.argmin(L, axis = 1)
    s = np.argmin(U, axis = 1)
    b = np.argmax(U, axis = 1)
   
    alpha = get_default_alpha_arr(K)
    learned_betas = []
    learned_predictions = []
    
    # Run code to compute the mixture iteratively
    for k in range(K):
        beta = cp.Variable((d,m))  # The parameters of the one-vs-all classifier

        # Solve relaxed convexified optimization problem

        # Constructing the optimization problem
        loss_objective = 0
        for i in range(n):
            # construct list with entries L(x_i, y) + beta_y^T x_i - beta_{y_i}^T x_i; for each y and y_i defined appropriately
            loss_objective += get_convex_version(X, L, beta, y, i)
            #loss_objective += cp.max(L[i,:] + cp.matmul(beta,X[i,:])) - cp.matmul(beta[y[i],:],X[i,:])

        constraints_obj = 0
        for i in range(n):
            # Computing the utility i has for their own assignments so far
            USFii = 0
            for t in range(k):
                USFii += alpha[t]*U[i, learned_predictions[t][i]]
            part2 = cp.max(-U[i,:] + beta*X[i,:]) - beta[b[i],:]*X[i,:]
            
            for j in range(n):
                if(i != j):
                    # Computing the utility j has for i's assignments so far
                    USFij = 0
                    for t in range(k):
                        USFij += alpha[t]*U[i, learned_predictions[t][j]]

                    # computing both parts of the constraint loss relaxed
                    part1 = cp.max(U[i,:] + beta*X[j,:]) - beta[s[i],:]*X[j,:]#beta[b[j],:]*X[j,:]
                    constraints_obj += cp.maximum(USFij+alpha[k]*part1 -USFii+alpha[k]*part2, 0)

        objective = cp.Minimize((1/n)*loss_objective + lamb*(1/(n*(n-1)))*constraints_obj)
        prob = cp.Problem(objective)

        # Solving the problem
        results = prob.solve(verbose=False)
        beta_value = np.array(beta.value)
        learned_betas.append(beta_value)

        all_predictions = predictions(beta_value, X)
        learned_predictions.append(all_predictions)

    alphas = cp.Variable(K)
    xis = cp.Variable((n,n))

    learned_pred_losses = []
    for k in range(K):
        this_loss = 0
        for i in range(n):
            this_loss += L[i, learned_predictions[k][i]]
        learned_pred_losses.append(this_loss)

    objective = cp.Minimize(cp.sum(cp.multiply(learned_pred_losses, alphas)) + lamb*cp.sum(xis))

    constraints = []
    for i in range(n):
        for j in range(n):
            if(j != i):
                util_diffs = []
                for k in range(K):
                    util_diffs.append(U[i, learned_predictions[k][i]] - U[i, learned_predictions[k][j]])
                constraints.append(cp.sum(cp.multiply(util_diffs, alphas)) + xis[i,j] >= 0)

    for i in range(n):
        for j in range(n):
            constraints.append(xis[i,j] >= 0)

    for k in range(K):
        constraints.append(alphas[k] >= 0)

    constraints.append(cp.sum(alphas) == 1)
    prob = cp.Problem(objective, constraints)
    results = prob.solve()
    opt_alphas = np.array(alphas.value).flatten()
    
    learned_pred_group = None
    if groups is not None and U_mat is not None:
        num_groups = len(groups.keys())
        learned_pred_group = {h:[] for h in range(num_groups)} 
        for k in range(K):
            for h in range(num_groups):
                learned_pred_group[h].append(predictions(learned_betas[k], groups[h]))
    
    return learned_betas, learned_predictions, learned_pred_group, opt_alphas
"""

def train_erm_envy_free(X, L_mat, U_mat, groups, lamb=0.5):
    L_X = np.matmul(X, L_mat.T)
    L_X = normalize(L_X, axis=1, norm='l1')

    U_X = np.matmul(X, U_mat.T)
    U_X = normalize(U_X, axis=1, norm='l1')
    
    n, d = L_X.shape
    n, m = X.shape
    
    # Constructing argmax/min labels used repeatedly
    y = np.argmin(L_X, axis = 1)
    s = np.argmin(U_X, axis = 1)
    b = np.argmax(U_X, axis = 1)

    learned_betas = []
    learned_predictions = {h:[] for h in range(num_groups)} 
    learned_predictions_all = []
    def_alphas = get_default_alpha_arr(K)
        
    # Run code to compute the mixture iteratively
    for k in range(K):
        Beta = cp.Variable((d,m))  # The parameters of the one-vs-all classifier
        
        # Solve relaxed convexified optimization problem
        loss_objective = 0
        for i in range(n):
            # construct list with entries L(x_i, y) + beta_y^T x_i - beta_{y_i}^T x_i; for each y and y_i defined appropriately
            loss_objective += get_convex_version(X, L_X, Beta, y, i)

        constraints_obj = 0
        for i in range(n):
            USFii = 0
            for t in range(k):
                USFii += def_alphas[t]*U_X[i, learned_predictions_all[t][i]]
            
            convex_version_ii = cp.max(-U_X[i, :] + cp.matmul(Beta,X[i,:])) - \
                    cp.matmul(Beta[b[i],:],X[i,:])
            convex_version_ii *= def_alphas[k]

            for j in range(n):
                if i != j:
                    USFij = 0
                    for t in range(k):
                        USFij += def_alphas[t]*U_X[i, learned_predictions_all[t][j]]
                    convex_version_ij = cp.max(U_X[i, :] + cp.matmul(Beta,X[j,:])) - cp.matmul(Beta[s[i],:],X[j,:])
                    convex_version_ij *= def_alphas[k]
                    constraints_obj += cp.maximum(USFij+convex_version_ij- USFii+convex_version_ii, 0)

        objective = cp.Minimize((1/n)*loss_objective + lamb*(1/(n*(n-1)))*constraints_obj)
        prob = cp.Problem(objective)

        # Solving the problem
        #try:
        results = prob.solve(solver=cp.SCS, verbose=False)#, feastol=1e-5, abstol=1e-5)
        #except:
        #    return 0, 0, 0, 0
        Beta_value = np.array(Beta.value)
        learned_betas.append(Beta_value)
    
        #total_envy = compute_envy_upper(Beta_value, k)
        #print("Total envy in training is: ", total_envy)

        all_predictions = predictions(Beta_value, X)
        learned_predictions_all.append(all_predictions)

        for h in range(num_groups):
            learned_predictions[h].append(predictions(Beta_value, groups[h]))

    # We now solve for the optimal alpha values
    alphas = cp.Variable(K)
    xis = cp.Variable((n,n))

    learned_pred_losses = []
    for k in range(K):
        this_loss = 0
        for i in range(n):
            this_loss += L_X[i, learned_predictions_all[k][i]]
        learned_pred_losses.append(this_loss)

    objective = cp.Minimize(cp.sum(cp.multiply(learned_pred_losses, alphas)) + lamb*cp.sum(xis))
    constraints = []
    for i in range(n):
        for j in range(n):
            if i != j:
                util_diffs = []
                for k in range(K):
                    util_diffs.append(U_X[i, learned_predictions_all[k][i]] - \
                            U_X[i, learned_predictions_all[k][j]])    
                constraints.append(cp.sum(cp.multiply(util_diffs, alphas)) + xis[i,j] >= 0)
    
    for i in range(n):
        for j in range(n):
            constraints.append(xis[i,j] >= 0)
    for k in range(K):
        constraints.append(alphas[k] >= 0)
    constraints.append(cp.sum(alphas) == 1)
    prob = cp.Problem(objective, constraints)
    try:
        results = prob.solve(cp.SCS, verbose=False)#, feastol=1e-5, abstol=1e-5)
    except:
        return 0,0,0,0 
    opt_alphas = np.array(alphas.value).flatten()
    
    return learned_betas, learned_predictions_all, learned_predictions, opt_alphas

def test_erm_envy_free():
    Lambda = 10
    print("Envy Free Test: ")
    print("First compute ERM solution: ")
    train_X = np.array([[0.8, 0.3, 1.5, 0.1], \
                        [0.3, 1.1, 1.7, 0.9], \
                        [1.1, 1.4, 0.5, 1.2],
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
    avg_envy, violations = total_average_envy(opt_alphas, U, train_X, learned_pred_all) 
    loss = compute_final_loss(opt_alphas, L, train_X, learned_pred_all)
    optimal_loss = get_optimal_loss(L, train_X)
    
    print("Optimal Loss is:", optimal_loss)
    print("ERM loss is: ", loss)
    print("ERM total envy: ", avg_envy, "violations: ", violations)
    print("")

    print("Now do wth Envy Free constraint with lambda: ", Lambda)
    learned_betas, learned_pred_all, learned_pred_group, opt_alphas = \
            train_erm_envy_free(train_X, L, U, samples_by_group, lamb=Lambda)
    avg_envy, violations = total_average_envy(opt_alphas, U, train_X, learned_pred_all) 
    loss = compute_final_loss(opt_alphas, L, train_X, learned_pred_all)
    print("ERM-Envy Free Loss is : ", loss)
    print("ERM-Envy Free avg envy: ", avg_envy, "violations: ", violations)

def size_test():
    n = 50
    m = 14
    d = 5
    K = 4
    print("\nRunning Size Test")

    train_X = generate_data(n, m, 'uniform')
    train_X = normalize(train_X, axis=1, norm='l1')    
    group_dist = [0.25, 0.25, 0.25, 0.25]
    samples_by_group = define_groups(train_X, group_dist)
    L = generate_loss_matrix(d, m, 'uniform')
    U = generate_utility_matrix_var(d, m, 'uniform', 0)

    erm_betas, learned_predictions, learned_pred_group, alphas = train_erm(train_X, L, U, \
            samples_by_group)
    final_loss = compute_final_loss(alphas, L, train_X, learned_predictions)
    avg_envy, violations = total_average_envy(alphas, U, train_X, learned_predictions) 
    opt_loss = get_optimal_loss(L, train_X)
    
    print("Optimal loss is: ", opt_loss)
    print("ERM loss is: ", final_loss)
    print("ERM average envy: ", avg_envy, "ERM total violations: ", violations)
   
    start_time = time.time()
    cons_betas, learned_predictions, learned_pred_group, opt_alphas = \
            train_erm_envy_free(train_X, L, U, samples_by_group, lamb=100)
    final_loss = compute_final_loss(opt_alphas, L, train_X, learned_predictions)
    avg_envy, violations = total_average_envy(opt_alphas, U, train_X, learned_predictions) 
    print("ERM-Envy Free loss is: ", final_loss)
    print("ERM-Envy Free average envy: ", avg_envy, "ERM total violations: ", violations)
    end_time = time.time()
    print("Time is: ", end_time - start_time)
    print(learned_predictions)

    test_X = generate_data(n, m, 'uniform')
    group_dist = [0.25, 0.25, 0.25, 0.25]
    test_s_by_group = define_groups(test_X, group_dist)
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(erm_betas, test_X, test_s_by_group, K)
    st_avg_envy, st_envy_violations = total_average_envy(opt_alphas, U, train_X, \
            st_learned_predictions)
    #print("ERM get this much envy on test: ", st_avg_envy, "Violations: ", st_envy_violations)
    
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(cons_betas, test_X, test_s_by_group, K)
    st_avg_envy, st_envy_violations = total_average_envy(opt_alphas, U, test_X, \
            st_learned_predictions)
    #print("ERM-GEF get this much envy on test: ", st_avg_envy, "Violations: ", st_envy_violations)
     
if __name__ == "__main__":
    test_erm_envy_free()
    size_test()
    #while(True):
    #    size_test()

