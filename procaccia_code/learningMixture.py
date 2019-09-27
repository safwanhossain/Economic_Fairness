from cvxpy import *
import numpy as np
from sys import argv
import time
import matplotlib.pyplot as plt

Lambda = float(argv[1])
K = int(argv[2])        # Number of functions in the mixture
folder_name = "Outputs/"

alpha = []
alpha_value = 1.0
for k in range(K-1):
        alpha_value = alpha_value / 2
        alpha.append(alpha_value)
alpha.append(alpha_value)

if argv[3] == "load":
        L_total = np.loadtxt('L.csv', delimiter=',')
        u_total = np.loadtxt('u.csv', delimiter=',')
        X_total = np.loadtxt('X.csv', delimiter=',')

        (n_total,m) = np.shape(L_total) # n_total is the number of users (train + test), and m is the number of alterantives
        (_,d) = np.shape(X_total)       # d is the number of features for each user (dimension of X)

else:
        n_total = int(argv[4])  # Total number of users, i.e. |V|
        m = int(argv[5])        # Number of alterantives, i.e. |A|
        d = int(argv[6])        # Number of features for each user (dimension of X)

        X_total = np.abs(np.random.randn(n_total,d))
        L_total = np.dot(X, 10*np.abs(np.random.randn(d,m)))    # L(x_i, y) = x_i^T H_y, i.e. loss depends linearly on features of individual (and corresponding y)
        u_total = np.dot(X, 10*np.abs(np.random.randn(d,m)))    # u(x_i, y) = x_i^T B_y, i.e. loss depends linearly on features of individual (and corresponding y)

# Making the train/test split
n_test = int(n_total / 4)       # Using quarter of the points as test points
n = n_total - n_test

X = X_total[:n, :]
X_test = X_total[n:, :]

L = L_total[:n, :]
L_test = L_total[n:, :]

u = u_total[:n, :]
u_test = u_total[n:, :]


def predictions(beta_value, all_X):
        return np.argmax(np.dot(beta_value, all_X.T), axis = 0)

print()
print("SOLVING TO COMPUTE MIXTURE COMPONENTS")

start_time = time.time()

# Constructing argmax/min labels used repeatedly
y = np.argmin(L, axis = 1)
s = np.argmin(u, axis = 1)
b = np.argmax(u, axis = 1)

# Storing betas learned so far and their predictions
learned_betas = []
learned_predictions = []

# Run code to compute the mixture iteratively
for k in range(K):
        beta = Variable((m,d))  # The parameters of the one-vs-all classifier

        # Solve relaxed convexified optimization problem

        # Constructing the optimization problem
        loss_objective = 0
        for i in range(n):
                # construct list with entries L(x_i, y) + beta_y^T x_i - beta_{y_i}^T x_i; for each y and y_i defined appropriately
                loss_objective += cvxpy.max(L[i,:] + beta*X[i,:]) - beta[y[i],:]*X[i,:]

        constraints_obj = 0
        for i in range(n):
                # Computing the utility i has for their own assignments so far
                USFii = 0
                for t in range(k):
                        USFii += alpha[t]*u[i, learned_predictions[t][i]]
                
                for j in range(n):
                        if(i != j):
                                # Computing the utility j has for i's assignments so far
                                USFij = 0
                                for t in range(k):
                                        USFij += alpha[t]*u[i, learned_predictions[t][j]]

                                # computing both parts of the constraint loss relaxed
                                part1 = cvxpy.max(u[i,:] + beta*X[j,:]) - beta[s[i],:]*X[j,:]#beta[b[j],:]*X[j,:]
                                part2 = cvxpy.max(-u[i,:] + beta*X[i,:]) - beta[b[i],:]*X[i,:]
                                constraints_obj += maximum(USFij+alpha[k]*part1 -USFii+alpha[k]*part2, 0)

        objective = Minimize((1/n)*loss_objective + Lambda*(1/(n*(n-1)))*constraints_obj)
        prob = Problem(objective)

        # Solving the problem
        results = prob.solve()
        beta_value = np.array(beta.value)
        learned_betas.append(beta_value)

        all_predictions = predictions(beta_value, X)
        learned_predictions.append(all_predictions)

end_time = time.time()

print()
print("Time taken for computing mixture:", end_time - start_time)
print()

final_loss = 0
for k in range(K):
        for i in range(n):
                final_loss += alpha[k]*L[i, learned_predictions[k][i]]

print("Average Loss on Training:", final_loss/n)
print()

violated_num = 0
total_constr = 0
envy_values = []
for i in range(n):
        for j in range(n):
                if(i != j):
                        total_constr += 1

                        utility_i = 0
                        utility_j = 0
                        for k in range(K):
                                utility_i += alpha[k]*u[i, learned_predictions[k][i]]
                                utility_j += alpha[k]*u[i, learned_predictions[k][j]]

                        if(utility_i < utility_j -1e-10):
                                violated_num += 1

                        envy_values.append(utility_j - utility_i)
envy_values = np.array(envy_values)

print(violated_num, "constraints of", total_constr, "violated for train")
print()

u_max = np.amax(u)
u_min = np.amin(u)

envy_max = u_max - u_min
envy_min = u_min - u_max

plt.hist(envy_values -1e-10, bins = 50, range = (envy_min,envy_max), edgecolor='black', linewidth=0.8)  # arguments are passed to np.histogram, using -1e-10 to include the right end of bin into bar
plt.title("Histogram of train envy values (before alpha opt)")
plt.savefig(folder_name+"TrainEnvyDist_mixture_beforeAlpha.png")
plt.clf()
np.savetxt(folder_name+"TrainEnvies_mixture_beforeAlpha.csv", envy_values, delimiter=',')


test_predictions = []
for k in range(K):
        test_predictions.append(predictions(learned_betas[k], X_test))

test_loss = 0
for k in range(K):
        for i in range(n_test):
                test_loss += alpha[k]*L_test[i, test_predictions[k][i]]

print("Average Loss on Test:", test_loss/n_test)
print()

violated_num = 0
total_constr = 0
envy_values = []
for i in range(n_test):
        for j in range(n_test):
                if(i != j):
                        total_constr += 1

                        utility_i = 0
                        utility_j = 0
                        for k in range(K):
                                utility_i += alpha[k]*u_test[i, test_predictions[k][i]]
                                utility_j += alpha[k]*u_test[i, test_predictions[k][j]]

                        if(utility_i < utility_j -1e-10):
                                violated_num += 1

                        envy_values.append(utility_j - utility_i)
envy_values = np.array(envy_values)

print(violated_num, "constraints of", total_constr, "violated for test")
print()

u_max = np.amax(u_test)
u_min = np.amin(u_test)

envy_max = u_max - u_min
envy_min = u_min - u_max

plt.hist(envy_values -1e-10, bins = 50, range = (envy_min,envy_max), edgecolor='black', linewidth=0.8)  # arguments are passed to np.histogram, using -1e-10 to include the right end of bin into bar
plt.title("Histogram of test envy values (after alpha opt)")
plt.savefig(folder_name+"TestEnvyDist_mixture_beforeAlpha.png")
plt.clf()
np.savetxt(folder_name+"TestEnvies_mixture_beforeAlpha.csv", envy_values, delimiter=',')


if argv[3] == 'gen':    # That is, not "load"ing the data
        np.savetxt("L.csv", L, delimiter=',')
        np.savetxt("u.csv", u, delimiter=',')
        np.savetxt("X.csv", X, delimiter=',')


# Adding a layer of solving for alpha to see if things improve

print()
print("SOLVING FOR COMPUTING OPTIMAL ALPHA")
print()

start_time = time.time()

alphas = Variable(K)
xis = Variable((n,n))

learned_pred_losses = []
for k in range(K):
        this_loss = 0
        for i in range(n):
                this_loss += L[i, learned_predictions[k][i]]
        learned_pred_losses.append(this_loss)

objective = Minimize(cvxpy.sum(multiply(learned_pred_losses, alphas)) + Lambda*cvxpy.sum(xis))

constraints = []
for i in range(n):
        for j in range(n):
                if(j != i):
                        util_diffs = []
                        for k in range(K):
                                util_diffs.append(u[i, learned_predictions[k][i]] - u[i, learned_predictions[k][j]])
                        
                        constraints.append(cvxpy.sum(multiply(util_diffs, alphas)) + xis[i,j] >= 0)

for i in range(n):
        for j in range(n):
                constraints.append(xis[i,j] >= 0)

for k in range(K):
        constraints.append(alphas[k] >= 0)

constraints.append(cvxpy.sum(alphas) == 1)

prob = Problem(objective, constraints)

results = prob.solve()
alpha = np.array(alphas.value).flatten()

end_time = time.time()

print("Time taken to compute alphas:", end_time - start_time)
print()

final_loss = 0
for k in range(K):
        for i in range(n):
                final_loss += alpha[k]*L[i, learned_predictions[k][i]]

print("Average Loss on Train:", final_loss/n)
print()

violated_num = 0
total_constr = 0
envy_values = []
for i in range(n):
        for j in range(n):
                if(i != j):
                        total_constr += 1

                        utility_i = 0
                        utility_j = 0
                        for k in range(K):
                                utility_i += alpha[k]*u[i, learned_predictions[k][i]]
                                utility_j += alpha[k]*u[i, learned_predictions[k][j]]

                        if(utility_i < utility_j -1e-10):
                                violated_num += 1

                        envy_values.append(utility_j - utility_i)
envy_values = np.array(envy_values)

print(violated_num, "constraints of", total_constr, "violated for train")
print()

u_max = np.amax(u)
u_min = np.amin(u)

envy_max = u_max - u_min
envy_min = u_min - u_max

plt.hist(envy_values -1e-10, bins = 50, range = (envy_min,envy_max), edgecolor='black', linewidth=0.8)  # arguments are passed to np.histogram, using -1e-10 to include the right end of bin into bar
plt.title("Histogram of train envy values")
plt.savefig(folder_name+"TrainEnvyDist_mixture_afterAlpha.png")
plt.clf()
np.savetxt(folder_name+"TrainEnvies_mixture_afterAlpha.csv", envy_values, delimiter=',')


test_loss = 0
for k in range(K):
        for i in range(n_test):
                test_loss += alpha[k]*L_test[i, test_predictions[k][i]]

print("Average Loss on Test:", test_loss/n_test)
print()

violated_num = 0
total_constr = 0
envy_values = []
for i in range(n_test):
        for j in range(n_test):
                if(i != j):
                        total_constr += 1

                        utility_i = 0
                        utility_j = 0
                        for k in range(K):
                                utility_i += alpha[k]*u_test[i, test_predictions[k][i]]
                                utility_j += alpha[k]*u_test[i, test_predictions[k][j]]

                        if(utility_i < utility_j -1e-10):
                                violated_num += 1

                        envy_values.append(utility_j - utility_i)
envy_values = np.array(envy_values)

print(violated_num, "constraints of", total_constr, "violated for test")
print()

u_max = np.amax(u_test)
u_min = np.amin(u_test)

envy_max = u_max - u_min
envy_min = u_min - u_max

plt.hist(envy_values -1e-10, bins = 50, range = (envy_min,envy_max), edgecolor='black', linewidth=0.8)  # arguments are passed to np.histogram, using -1e-10 to include the right end of bin into bar
plt.title("Histogram of test envy values")
plt.savefig(folder_name+"TestEnvyDist_mixture_afterAlpha.png")
plt.clf()
np.savetxt(folder_name+"TestEnvies_mixture_afterAlpha.csv", envy_values, delimiter=',')
