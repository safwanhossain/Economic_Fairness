import numpy as np
from sys import argv
from cvxpy import *
import time

n_train = int(argv[1])	# Number of users to use in training data
n = n_train + int(n_train / 3)	# Generate an additional 33% of training as test data
m = int(argv[2])	# Number of outcomes
d = int(argv[3])	# Number of features for each user
K = int(argv[4])	# Number of functions in the mixture
folder_name = "Outputs/"

def get_predictions(beta_value, all_X):
	return np.argmax(np.dot(beta_value, all_X.T), axis = 0)

def get_prediction(beta_value, x_value):
	return np.argmax(np.dot(beta_value, x_value), axis = 0)

start_time = time.time()

betas = []	# Parameters of each of the classifiers of the mixture
for k in range(K):
	betas.append(np.random.randn(m, d))

alphas = np.random.uniform(size = K)	# Weight/probability of each classifier
alphas = alphas / (np.sum(alphas))


# Generate the feature matrix
X = np.random.uniform(low=0.0, high=1.0, size=(n,d))

all_predictions = []
for k in range(K):
	all_predictions.append(get_predictions(betas[k], X))
all_predictions = np.array(all_predictions, dtype = int)

# Generate a loss function with zero loss for this mixture, and maximum loss otherwise
L = np.random.uniform(low=0.0, high=1.0, size=(n,m))
for i in range(n):
	this_preds = all_predictions[:,i]
	L[i, this_preds] = 0

# Generate a utility function consistent (wrt EF) with the mixture of classifiers, via an LP
u = Variable((n,m))
objective = Minimize(0)
constraints = []
for i in range(n):
	for j in range(n):
		if(i != j):
			constraints.append(cvxpy.sum(cvxpy.multiply(alphas, u[i, all_predictions[:,i]] - u[i, all_predictions[:,j]])) >= 0.00001)
#for i in range(n):
#    constraints.append(cvxpy.pnorm(u[i,:], p=2) == 1)

prob = Problem(objective, constraints)
prob.solve()
u = np.array(u.value)
print(u)
u = u - np.amin(u)	# So that the minimum is 0
u = u / np.amax(u)	# To bound to 1

# Randomizing the users, to make sure train-test split is indeed random
rand_perm = np.random.permutation(range(n))
X = X[rand_perm, :]
L = L[rand_perm, :]
u = u[rand_perm, :]

end_time = time.time()

print()
print("Time taken for generation:", end_time - start_time)


all_predictions = []
for k in range(K):
	all_predictions.append(get_predictions(betas[k], X))
all_predictions = np.array(all_predictions, dtype = int)


final_loss = 0
for k in range(K):
	for i in range(n):
		final_loss += alphas[k]*L[i, all_predictions[k,i]]

print()
print("Loss objective value:", final_loss)

violated_num = 0
total_constr = n_train*(n_train-1)
envy_values = []
for i in range(n_train):
	for j in range(n_train):
		if(i != j):
			utility_i = 0	# util of i for f(x_i)
			utility_j = 0	# util of i for f(x_j)
			for k in range(K):
				utility_i += alphas[k]*u[i, all_predictions[k,i]]
				utility_j += alphas[k]*u[i, all_predictions[k,j]]

			if(utility_i < utility_j -1e-10):
				violated_num += 1

			envy_values.append(utility_j - utility_i)
envy_values = np.array(envy_values)
print("U MATRIX")
print(u)

print()
print(violated_num, "constraints of", total_constr, "violated for train")

import matplotlib.pyplot as plt
plt.hist(envy_values -1e-10, bins = 50, range = (-1,1), edgecolor='black', linewidth=0.8)  # arguments are passed to np.histogram
plt.title("Histogram of train envy values for optimal")
plt.savefig(folder_name+"TrainEnvyDist_optimal.png")
plt.clf()
np.savetxt(folder_name+"TrainEnvies_optimal.csv", envy_values, delimiter=',')


n_test = int(n_train / 3)
violated_num = 0
total_constr = n_test*(n_test-1)
envy_values = []
for i in range(n_train, n):
	for j in range(n_train, n):
		if(i != j):
			utility_i = 0	# util of i for f(x_i)
			utility_j = 0	# util of i for f(x_j)
			for k in range(K):
				utility_i += alphas[k]*u[i, all_predictions[k,i]]
				utility_j += alphas[k]*u[i, all_predictions[k,j]]

			if(utility_i < utility_j -1e-10):
				violated_num += 1

			envy_values.append(utility_j - utility_i)
envy_values = np.array(envy_values)

print()
print(violated_num, "constraints of", total_constr, "violated for test")
print()

plt.hist(envy_values -1e-10, bins = 50, range = (-1,1), edgecolor='black', linewidth=0.8)  # arguments are passed to np.histogram
plt.title("Histogram of test envy values for optimal")
plt.savefig(folder_name+"TestEnvyDist_optimal.png")
plt.clf()
np.savetxt(folder_name+"TestEnvies_optimal.csv", envy_values, delimiter=',')


np.savetxt("L.csv", L, delimiter=',')
np.savetxt("u.csv", u, delimiter=',')
np.savetxt("X.csv", X, delimiter=',')
