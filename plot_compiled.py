from sys import argv
import os
import numpy as np
import matplotlib.pyplot as plt

runMax = int(argv[1])
mode = 'plot'

if len(argv) >= 3:
	mode = argv[2]

if 'gen' in mode:
	for r in range(1, runMax+1):
		os.system("python compile_results.py " + str(r) + " compile")

time_datas = []

for r in range(1, runMax+1):
	time_datas.append(np.loadtxt("Run" + str(r) + "/times.csv", delimiter=','))

time_datas = np.array(time_datas)
time_data = np.mean(time_datas, axis = 0)

plt.plot(time_data[:,0], time_data[:,1], linewidth = 3, color = '#0089D0', marker = 'x', label = 'mixture computation')
plt.plot(time_data[:,0], time_data[:,2], linewidth = 3, color = '#F37021', marker = 'x', label = 'eta computation')
plt.xlabel('Number of train individuals',fontsize=20, labelpad = 10)
plt.tick_params('x', labelsize=15, pad=10)
plt.ylabel('Time (sec)', fontsize=20, labelpad = 5)
plt.tick_params('y', labelsize=15)
plt.legend(fontsize = 10)
# plt.set_ylim(0.0, 1750.0)
plt.tight_layout()
plt.savefig('times.pdf')
plt.clf()


loss_datas = []

for r in range(1, runMax+1):
	loss_datas.append(np.loadtxt("Run" + str(r) + "/losses.csv", delimiter=','))

loss_datas = np.array(loss_datas)
loss_data = np.mean(loss_datas, axis = 0)

plt.plot(loss_data[:,0], loss_data[:,1], linewidth = 3, color = '#6460AA', marker = 'x', label = 'train loss after mixture')
plt.plot(loss_data[:,0], loss_data[:,2], linewidth = 3, color = '#F37021', marker = 'x', label = 'train loss after eta')
plt.plot(loss_data[:,0], loss_data[:,3], linewidth = 3, color = '#0089D0', marker = 'x', label = 'test loss after mixture')
plt.plot(loss_data[:,0], loss_data[:,4], linewidth = 3, color = '#CC004C', marker = 'x', label = 'test loss after eta')
plt.plot(loss_data[:,0], loss_data[:,5], linewidth = 1, color = '#0DB14B', marker = 'x', linestyle = 'dashed', label = 'random assignment')
plt.xlabel('Number of train individuals',fontsize=20, labelpad = 10)
plt.tick_params('x', labelsize=15, pad=10)
plt.ylabel('Average loss', fontsize=20, labelpad = 5)
plt.tick_params('y', labelsize=15)
plt.legend(fontsize = 10)
# plt.set_ylim(0.0, 1750.0)
plt.tight_layout()
plt.savefig('losses.pdf')
plt.clf()


envies_datas = []

for r in range(1, runMax+1):
	envies_datas.append(np.loadtxt("Run" + str(r) + "/envies.csv", delimiter=','))

envies_datas = np.array(envies_datas)
envies_data = np.mean(envies_datas, axis = 0)

plt.plot(envies_data[:,0], envies_data[:,1], linewidth = 3, color = '#6460AA', marker = 'x', label = 'train envy after mixture')
plt.plot(envies_data[:,0], envies_data[:,2], linewidth = 3, color = '#F37021', marker = 'x', label = 'train envy after eta')
plt.plot(envies_data[:,0], envies_data[:,3], linewidth = 3, color = '#0089D0', marker = 'x', label = 'test envy after mixture')
plt.plot(envies_data[:,0], envies_data[:,4], linewidth = 3, color = '#CC004C', marker = 'x', label = 'test envy after eta')
plt.plot(envies_data[:,0], envies_data[:,5], linewidth = 1, color = '#0DB14B', marker = 'x', linestyle = 'dashed', label = 'random assignment')
plt.xlabel('Number of train individuals',fontsize=20, labelpad = 10)
plt.tick_params('x', labelsize=15, pad=10)
plt.ylabel('Average clipped envy', fontsize=20, labelpad = 5)
plt.tick_params('y', labelsize=15)
plt.legend(fontsize = 10)
# plt.set_ylim(0.0, 1750.0)
plt.tight_layout()
plt.savefig('envies.pdf')
plt.clf()


envyCDF_datas = []

for r in range(1, runMax+1):
	envyCDF_datas.append(np.loadtxt("Run" + str(r) + "/envyCDF.csv", delimiter=','))

envyCDF_datas = np.array(envyCDF_datas)
envyCDF_data = np.mean(envyCDF_datas, axis = 0)

plt.axvline(x=0.0, color = '#dadada', linewidth = 0.25)
plt.plot(envyCDF_data[:,0], envyCDF_data[:,1], linewidth = 1.5, color = '#6460AA', label = 'train envy after mixture')
plt.plot(envyCDF_data[:,0], envyCDF_data[:,2], linewidth = 1.5, color = '#F37021', label = 'train envy after eta')
plt.plot(envyCDF_data[:,0], envyCDF_data[:,3], linewidth = 1.5, color = '#0089D0', label = 'test envy after mixture')
plt.plot(envyCDF_data[:,0], envyCDF_data[:,4], linewidth = 1.5, color = '#CC004C', label = 'test envy after eta')
plt.plot(envyCDF_data[:,0], envyCDF_data[:,5], linewidth = 1, color = '#FCB711', linestyle = 'dashed', label = 'envy by optimal')
plt.plot(envyCDF_data[:,0], envyCDF_data[:,6], linewidth = 1, color = '#0DB14B', linestyle = 'dashed', label = 'envy by random')
plt.xlabel('Value of envy',fontsize=20, labelpad = 10)
plt.tick_params('x', labelsize=15, pad=10)
plt.ylabel('Fractions of pairs', fontsize=20, labelpad = 5)
plt.tick_params('y', labelsize=15)
plt.legend(fontsize = 10)
plt.xlim(-0.75, 0.75)
plt.tight_layout()
plt.savefig('envyCDF_100.pdf')

np.savetxt('envyCDF_100_mean.csv', envyCDF_data, delimiter=',')
