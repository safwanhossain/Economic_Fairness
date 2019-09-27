import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import os

if len(argv) < 2:
	run = 1
else:
	run = int(argv[1])

if len(argv) < 3:
	mode = 'plot'
else:
	mode = argv[2]	# plot vs compile/save

all_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
Sizes = []	# sizes that succeeded/were valid

for n in all_sizes:
	files_in_folder = os.listdir("Run" + str(run) + "/Size" + str(n))
	if len(files_in_folder) > 9:
		Sizes.append(n)

if(len(all_sizes) != len(Sizes)):
	print("Warning! Run " + str(run) + " does not have all sizes succeeded")

Sizes = np.array(Sizes, dtype = int)

mixture_times = []
mixture_train = []
mixture_test = []

alpha_times = []
alpha_train = []
alpha_test = []

random_losses = []
random_envyValues = {}

for n in Sizes:
	f = open("Run" + str(run) + "/Size" + str(n) + "/learning_output.txt")
	lines = f.readlines()

	time = float(lines[3].split(":")[1])
	mixture_times.append(time)

	train_loss = float(lines[5].split(":")[1])
	mixture_train.append(train_loss)

	test_loss = float(lines[9].split(":")[1])
	mixture_test.append(test_loss)

	time = float(lines[16].split(":")[1])
	alpha_times.append(time)

	train_loss = float(lines[18].split(":")[1])
	alpha_train.append(train_loss)

	test_loss = float(lines[22].split(":")[1])
	alpha_test.append(test_loss)

	# Computing the loss & envies obtained by a random assignment
	L_total = np.loadtxt("Run" + str(run) + "/Size" + str(n) + '/L.csv', delimiter=',')
	(n_total, m) = np.shape(L_total)
	random_asgn = np.random.randint(m, size=n_total)
	this_loss = 0
	for i in range(n_total):
		this_loss += L_total[i, random_asgn[i]]
	random_losses.append(this_loss / n_total)

	u_total = np.loadtxt("Run" + str(run) + "/Size" + str(n) + '/u.csv', delimiter=',')
	n_train = n_total - int(n_total / 4)	# Training data size

	if n != n_train:
		print("Something is wrong!")
		exit(0)

	envy_values = []
	for i in range(n_train):
		for j in range(n_train):
			if(i != j):
				envy_values.append(u_total[i, random_asgn[j]] - u_total[i, random_asgn[i]])
	for i in range(n_train, n_total):
		for j in range(n_train, n_total):
			if(i != j):
				envy_values.append(u_total[i, random_asgn[j]] - u_total[i, random_asgn[i]])
	random_envyValues[n] = np.array(envy_values)


mixture_times = np.array(mixture_times)
mixture_train = np.array(mixture_train)
mixture_test = np.array(mixture_test)

alpha_times = np.array(alpha_times)
alpha_train = np.array(alpha_train)
alpha_test = np.array(alpha_test)

random_losses = np.array(random_losses)


if mode == 'plot':
	plt.plot(Sizes, mixture_times, linewidth = 3, color = 'blue', marker = 'x', label = 'mixture computation')
	plt.plot(Sizes, alpha_times, linewidth = 3, color = 'red', marker = 'x', label = 'eta computation')
	plt.xlabel('Number of train individuals',fontsize=20, labelpad = 10)
	plt.tick_params('x', labelsize=15, pad=10)
	plt.ylabel('Time (sec)', fontsize=20, labelpad = 5)
	plt.tick_params('y', labelsize=15)
	plt.legend(fontsize = 10)
	# plt.set_ylim(0.0, 1750.0)
	plt.tight_layout()
	plt.savefig('times.pdf')
	plt.clf()

else:
	times_data = np.vstack((Sizes, mixture_times, alpha_times)).T
	np.savetxt('Run' + str(run) + '/times.csv', times_data, delimiter = ',')


if mode == 'plot':
	plt.plot(Sizes, mixture_train, linewidth = 3, color = 'green', marker = 'x', label = 'train loss after mixture')
	plt.plot(Sizes, alpha_train, linewidth = 3, color = 'maroon', marker = 'x', label = 'train loss after eta')
	plt.plot(Sizes, mixture_test, linewidth = 3, color = 'blue', marker = 'x', label = 'test loss after mixture')
	plt.plot(Sizes, alpha_test, linewidth = 3, color = 'red', marker = 'x', label = 'test loss after eta')
	plt.plot(Sizes, random_losses, linewidth = 1, color = 'orange', marker = 'x', linestyle = 'dashed', label = 'random assignment')
	plt.xlabel('Number of train individuals',fontsize=20, labelpad = 10)
	plt.tick_params('x', labelsize=15, pad=10)
	plt.ylabel('Average loss', fontsize=20, labelpad = 5)
	plt.tick_params('y', labelsize=15)
	plt.legend(fontsize = 10)
	# plt.set_ylim(0.0, 1750.0)
	plt.tight_layout()
	plt.savefig('losses.pdf')
	plt.clf()

else:
	losses_data = np.vstack((Sizes, mixture_train, alpha_train, mixture_test, alpha_test, random_losses)).T
	np.savetxt('Run' + str(run) + '/losses.csv', losses_data, delimiter = ',')


mixture_trainEnvy = []
mixture_testEnvy = []

alpha_trainEnvy = []
alpha_testEnvy = []

random_envy = []

for n in Sizes:
	train_envies = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TrainEnvies_mixture_beforeAlpha.csv", delimiter = ",")
	train_envies = np.array([max(e,0) for e in train_envies])
	mixture_trainEnvy.append(np.mean(train_envies))	# Average envy between pairs

	test_envies = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TestEnvies_mixture_beforeAlpha.csv", delimiter = ",")
	test_envies = np.array([max(e,0) for e in test_envies])
	mixture_testEnvy.append(np.mean(test_envies))	# Average envy between pairs

	train_envies = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TrainEnvies_mixture_afterAlpha.csv", delimiter = ",")
	train_envies = np.array([max(e,0) for e in train_envies])
	alpha_trainEnvy.append(np.mean(train_envies))	# Average envy between pairs

	test_envies = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TestEnvies_mixture_afterAlpha.csv", delimiter = ",")
	test_envies = np.array([max(e,0) for e in test_envies])
	alpha_testEnvy.append(np.mean(test_envies))	# Average envy between pairs

	random_envies = random_envyValues[n]
	random_envies = np.array([max(e,0) for e in random_envies])
	random_envy.append(np.mean(random_envies))

mixture_trainEnvy = np.array(mixture_trainEnvy)
mixture_testEnvy = np.array(mixture_testEnvy)

alpha_trainEnvy = np.array(alpha_trainEnvy)
alpha_testEnvy = np.array(alpha_testEnvy)

random_envy = np.array(random_envy)


if mode == 'plot':
	plt.plot(Sizes, mixture_trainEnvy, linewidth = 3, color = 'green', marker = 'x', label = 'train envy after mixture')
	plt.plot(Sizes, alpha_trainEnvy, linewidth = 3, color = 'maroon', marker = 'x', label = 'train envy after eta')
	plt.plot(Sizes, mixture_testEnvy, linewidth = 3, color = 'blue', marker = 'x', label = 'test envy after mixture')
	plt.plot(Sizes, alpha_testEnvy, linewidth = 3, color = 'red', marker = 'x', label = 'test envy after eta')
	plt.plot(Sizes, random_envy, linewidth = 1, color = 'orange', marker = 'x', linestyle = 'dashed', label = 'random assignment')
	plt.xlabel('Number of train individuals',fontsize=20, labelpad = 10)
	plt.tick_params('x', labelsize=15, pad=10)
	plt.ylabel('Average clipped envy', fontsize=20, labelpad = 5)
	plt.tick_params('y', labelsize=15)
	plt.legend(fontsize = 10)
	# plt.set_ylim(0.0, 1750.0)
	plt.tight_layout()
	plt.savefig('envies.pdf')
	plt.clf()

else:
	envies_data = np.vstack((Sizes, mixture_trainEnvy, alpha_trainEnvy, mixture_testEnvy, alpha_testEnvy, random_envy)).T
	np.savetxt('Run' + str(run) + '/envies.csv', envies_data, delimiter = ',')


# Computing the "CDF" of envy values at 100 training individuals
n = 100

if n in Sizes:
	train_envies_mixture = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TrainEnvies_mixture_beforeAlpha.csv", delimiter = ",")
	test_envies_mixture = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TestEnvies_mixture_beforeAlpha.csv", delimiter = ",")

	train_envies_alpha = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TrainEnvies_mixture_afterAlpha.csv", delimiter = ",")
	test_envies_alpha = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TestEnvies_mixture_afterAlpha.csv", delimiter = ",")

	train_envies_opt = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TrainEnvies_optimal.csv", delimiter = ",")
	test_envies_opt = np.loadtxt("Run" + str(run) + "/Size" + str(n) + "/TestEnvies_optimal.csv", delimiter = ",")
	envies_opt = np.concatenate((train_envies_opt, test_envies_opt))

	envies_random = random_envyValues[100]


	train_pairNum = len(train_envies_mixture)
	test_pairNum = len(test_envies_mixture)
	total_pairNum = len(envies_random)

	if(train_pairNum + test_pairNum != total_pairNum):
		print("Something is wrong!")
		exit(0)

	(train_binned_mixture, bin_edges) = np.histogram(train_envies_mixture -1e-10, bins = 200, range = (-1,1))
	(test_binned_mixture, _) = np.histogram(test_envies_mixture -1e-10, bins = 200, range = (-1,1))
	(train_binned_alpha, _) = np.histogram(train_envies_alpha -1e-10, bins = 200, range = (-1,1))
	(test_binned_alpha, _) = np.histogram(test_envies_alpha -1e-10, bins = 200, range = (-1,1))
	(binned_opt, _) = np.histogram(envies_opt -1e-10, bins = 200, range = (-1,1))
	(binned_random, _) = np.histogram(envies_random -1e-10, bins = 200, range = (-1,1))

	if mode == 'plot':
		plt.plot(bin_edges[1:], np.cumsum(train_binned_mixture)/train_pairNum, linewidth = 1.5, color = 'green', label = 'train envy after mixture')
		plt.plot(bin_edges[1:], np.cumsum(train_binned_alpha)/train_pairNum, linewidth = 1.5, color = 'maroon', label = 'train envy after eta')
		plt.plot(bin_edges[1:], np.cumsum(test_binned_mixture)/test_pairNum, linewidth = 1.5, color = 'blue', label = 'test envy after mixture')
		plt.plot(bin_edges[1:], np.cumsum(test_binned_alpha)/test_pairNum, linewidth = 1.5, color = 'red', label = 'test envy after eta')
		plt.plot(bin_edges[1:], np.cumsum(binned_opt)/total_pairNum, linewidth = 1, color = 'black', linestyle = 'dashed', label = 'envy by optimal')
		plt.plot(bin_edges[1:], np.cumsum(binned_random)/total_pairNum, linewidth = 1, color = 'orange', linestyle = 'dashed', label = 'envy by random')
		plt.axvline(x=0.0, color = 'grey', linewidth = 0.5)
		plt.xlabel('Value of envy',fontsize=20, labelpad = 10)
		plt.tick_params('x', labelsize=15, pad=10)
		plt.ylabel('Fractions of pairs', fontsize=20, labelpad = 5)
		plt.tick_params('y', labelsize=15)
		plt.legend(fontsize = 10)
		plt.xlim(-0.75, 0.75)
		plt.tight_layout()
		plt.savefig('envyCDF_100.pdf')

	else:
		envyCDF_data = np.vstack((bin_edges[1:], np.cumsum(train_binned_mixture)/train_pairNum, np.cumsum(train_binned_alpha)/train_pairNum, np.cumsum(test_binned_mixture)/test_pairNum, np.cumsum(test_binned_alpha)/test_pairNum, np.cumsum(binned_opt)/total_pairNum, np.cumsum(binned_random)/total_pairNum)).T
		np.savetxt('Run' + str(run) + '/envyCDF.csv', envyCDF_data, delimiter = ',')
