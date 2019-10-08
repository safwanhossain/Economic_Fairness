import cvxpy as cp
import numpy as np
import time
from constants import *
from helpers import *
from erm_training import train_erm
from sklearn.preprocessing import normalize

def train_erm_gef(X, L_mat, U_mat, groups, lamb=0.5):
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
   
    def compute_envy_upper(given_Beta, k):
        constraints_obj=0
        for i in range(num_groups):
            # First compute the utility group i has for itself
            USFii = 0
            curr_utl_ii = 0
            for t in range(k):
                for l in range(group_sizes[i]):
                    USFii += def_alphas[t]*group_UX[i][l, learned_predictions[i][t][l]]
                    
            for l in range(group_sizes[i]):
                convex_version = np.max(-group_UX[i][l, :] + np.matmul(given_Beta,groups[i][l,:])) - \
                        np.matmul(given_Beta[b[i][l],:],groups[i][l,:])
                curr_utl_ii += def_alphas[k]*convex_version
            USFii = USFii * (1/group_sizes[i])
            curr_utl_ii = curr_utl_ii * (1/group_sizes[i])

            for j in range(num_groups):
                if i != j:
                    USFij = 0
                    for t in range(k):
                        for li in range(group_sizes[i]):
                            for lj in range(group_sizes[j]):
                                USFij += def_alphas[t]*group_UX[i][li, learned_predictions[j][t][lj]]
                    
                    curr_utl_ij = 0
                    for li in range(group_sizes[i]):
                        for lj in range(group_sizes[j]):
                            convex_version = np.max(group_UX[i][li,:] + np.matmul(given_Beta,groups[j][lj,:])) - \
                                    np.matmul(given_Beta[s[i][li],:],groups[j][lj,:])
                            curr_utl_ij += def_alphas[k]*convex_version

                    USFij = USFij / (group_sizes[j]*group_sizes[i])
                    curr_utl_ij = curr_utl_ij / (group_sizes[i]*group_sizes[j])
                   
                    constraints_obj += max(USFij + curr_utl_ij - USFii + curr_utl_ii, 0)
        
        return constraints_obj

    # Run code to compute the mixture iteratively
    for k in range(K):
        Beta = cp.Variable((d,m))  # The parameters of the one-vs-all classifier
        
        # Solve relaxed convexified optimization problem
        loss_objective = 0
        for i in range(n):
            # construct list with entries L(x_i, y) + beta_y^T x_i - beta_{y_i}^T x_i; for each y and y_i defined appropriately
            loss_objective += get_convex_version(X, L_X, Beta, y, i)

        # Our Envy-Free Objective is over groups - so iterate over them
        constraints_obj = 0
        for i in range(num_groups):
            # First compute the utility group i has for itself
            USFii = 0
            curr_utl_ii = 0
            for t in range(k):
                for l in range(group_sizes[i]):
                    USFii += def_alphas[t]*group_UX[i][l, learned_predictions[i][t][l]]
                    
            for l in range(group_sizes[i]):
                convex_version = cp.max(-group_UX[i][l, :] + cp.matmul(Beta,groups[i][l,:])) - \
                        cp.matmul(Beta[b[i][l],:],groups[i][l,:])
                curr_utl_ii += def_alphas[k]*convex_version
            USFii = USFii * (1/group_sizes[i])
            curr_utl_ii = curr_utl_ii * (1/group_sizes[i])

            for j in range(num_groups):
                if i != j:
                    USFij = 0
                    for t in range(k):
                        for li in range(group_sizes[i]):
                            for lj in range(group_sizes[j]):
                                USFij += def_alphas[t]*group_UX[i][li, learned_predictions[j][t][lj]]
                    
                    curr_utl_ij = 0
                    for li in range(group_sizes[i]):
                        for lj in range(group_sizes[j]):
                            convex_version = cp.max(group_UX[i][li,:] + cp.matmul(Beta,groups[j][lj,:])) - \
                                    cp.matmul(Beta[s[i][li],:],groups[j][lj,:])
                            curr_utl_ij += def_alphas[k]*convex_version

                    USFij = USFij / (group_sizes[j]*group_sizes[i])
                    curr_utl_ij = curr_utl_ij / (group_sizes[i]*group_sizes[j])
                    
                    constraints_obj += cp.maximum(USFij + curr_utl_ij - USFii + curr_utl_ii, 0)
    
        objective = cp.Minimize((1/10)*((1/n)*loss_objective + lamb*constraints_obj))
        prob = cp.Problem(objective)

        # Solving the problem
        try:
            results = prob.solve(solver=cp.SCS, verbose=False)#, feastol=1e-5, abstol=1e-5)
        except:
            return 0, 0, 0, 0
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
    xis = cp.Variable((num_groups,num_groups))

    learned_pred_losses = []
    for k in range(K):
        this_loss = 0
        for i in range(n):
            this_loss += L_X[i, learned_predictions_all[k][i]]
        learned_pred_losses.append(this_loss)

    objective = cp.Minimize(cp.sum(cp.multiply(learned_pred_losses, alphas)) + lamb*cp.sum(xis))
    constraints = []
    for i in range(num_groups):
        for j in range(num_groups):
            if i != j:
                util_diff = []
                for k in range(K):
                    total_ii_utl = 0
                    total_ij_utl = 0
                    for li in range(group_sizes[i]):
                        total_ii_utl += group_UX[i][li, learned_predictions[i][k][li]]   
                        for lj in range(group_sizes[j]):
                            total_ij_utl += group_UX[i][li, learned_predictions[j][k][lj]]   
                    diff = total_ii_utl/group_sizes[i] - total_ij_utl/(group_sizes[i]*group_sizes[j])
                    util_diff.append(diff)
                #print("util_diff", util_diff)
                #print("alphas: ", alphas)
                #print("xis: ", xis)
                constraints.append(cp.sum(cp.multiply(util_diff, alphas)) + xis[i,j] >= 0)
    
    for i in range(num_groups):
        for j in range(num_groups):
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

def test_erm_gef():
    Lambda = 10
    print("Group Envy Free Test: ")
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
    total_envy, violations = total_group_envy(opt_alphas, U, samples_by_group, learned_pred_group) 
    loss = compute_final_loss(opt_alphas, L, train_X, learned_pred_all)
    optimal_loss = get_optimal_loss(L, train_X)
    
    print("Optimal Loss is:", optimal_loss)
    print("ERM loss is: ", loss)
    print("ERM total envy: ", total_envy, "ERM total violations: ", violations)
    print("")

    print("Now do wth Envy Free constraint with lambda: ", Lambda)
    learned_betas, learned_pred_all, learned_pred_group, opt_alphas = \
            train_erm_gef(train_X, L, U, samples_by_group, lamb=Lambda)
    total_envy, violations = total_group_envy(opt_alphas, U, samples_by_group, learned_pred_group) 
    loss = compute_final_loss(opt_alphas, L, train_X, learned_pred_all)
    print("ERM-GEF Loss is : ", loss)
    print("ERM-GEF total envy: ", total_envy, "ERM-GEF total violations: ", violations)

def size_test():
    n = 50
    m = 16
    d = 5
    K = 4
    print("\nRunning Size Test")

    train_X = generate_data(n, m, 'uniform')
    group_dist = [0.25, 0.25, 0.25, 0.25]
    samples_by_group = define_groups(train_X, group_dist, True)
    L = generate_loss_matrix(d, m, 'uniform')
    U = generate_utility_matrix_var(d, m, 'uniform', 0)
    
    erm_betas, learned_predictions, learned_pred_group, alphas = train_erm(train_X, L, U, \
            samples_by_group)
    final_loss = compute_final_loss(alphas, L, train_X, learned_predictions)
    total_envy, violations = total_group_envy(alphas, U, samples_by_group, learned_pred_group) 
    opt_loss = get_optimal_loss(L, train_X)
    
    print("Optimal loss is: ", opt_loss)
    print("ERM loss is: ", final_loss)
    print("ERM total envy: ", total_envy, "ERM total violations: ", violations)
   
    start_time = time.time()
    cons_betas, learned_predictions, learned_pred_group, opt_alphas = \
            train_erm_gef(train_X, L, U, samples_by_group, lamb=10)
    final_loss = compute_final_loss(opt_alphas, L, train_X, learned_predictions)
    total_envy, violations = total_group_envy(alphas, U, samples_by_group, learned_pred_group) 
    print("ERM-GEF loss is: ", final_loss)
    print("ERM-GEF total envy: ", total_envy, "ERM total violations: ", violations)
    end_time = time.time()
    print("Time is: ", end_time - start_time)

    test_X = generate_data(n, m, 'uniform')
    group_dist = [0.25, 0.25, 0.25, 0.25]
    test_s_by_group = define_groups(test_X, group_dist, True)
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(erm_betas, test_X, test_s_by_group, K)
    st_total_envy, st_envy_violations = total_group_envy(opt_alphas, U, test_s_by_group, \
            st_learned_pred_group)
    print("ERM get this much envy on test: ", st_total_envy)
    
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(cons_betas, test_X, test_s_by_group, K)
    st_total_envy, st_envy_violations = total_group_envy(opt_alphas, U, test_s_by_group, \
            st_learned_pred_group)
    print("ERM-GEF get this much envy on test: ", st_total_envy)
     
if __name__ == "__main__":
    test_erm_gef()
    size_test()
