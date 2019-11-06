#!/usr/bin/python3
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from constants import *

def read_from_csv(filenames, indicies):
    """ Indicies must be 4 elements """
    f_x_vals, f_train, f_test, f_train_var, f_test_var = [], [], [], [], []
    for i, filename in enumerate(filenames):
        x_vals, train, train_var, test, test_var = np.array([]), np.array([]), \
                np.array([]), np.array([]), np.array([])
        csv_file = open(filename, mode='r')
        csv_reader = csv.reader(csv_file, delimiter=",")
        
        for row in csv_reader:
            # Confidence interval is computed according to: scale*(std_dev/sqrt(num_sims))
            # 90% confidence - scale of 1.6
            scale = 1.5
            x_vals = np.append(x_vals, float(row[0]))
            train = np.append(train, float(row[indicies[0]]))
            train_var = np.append(train_var, scale*(float(row[indicies[1]])/np.sqrt(num_sims)))
            test = np.append(test, float(row[indicies[2]]))
            test_var = np.append(test_var, scale*(float(row[indicies[3]])/np.sqrt(num_sims)))
   
        f_x_vals.append(x_vals)  
        f_train.append(train)  
        f_train_var.append(train_var)  
        f_test.append(test)  
        f_test_var.append(test_var)  
    
    return f_x_vals, f_train, f_train_var, f_test, f_test_var

def plot(labels, data, plot_var=False):
    """ Plot the test and train losses and fairness
        labels are tuple of the following: (plt_name, x_label, fairness_label)
        data is a tuple of: (x_vals, train, train_var, test_loss, test_loss_var, train_fairness, 
            train_fairness_var, test_fairness, test_fairness_var)
    """
    plt_name, x_label, fairness_label = labels
    f_x_vals, f_train, f_train_var, f_test, f_test_var = data
    fig, ax1 = plt.subplots()
    num_files = len(f_train)
    
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    #colors = ['tab:red', 'tab:green', 'tab:grey']
    #colors = ['tab:green', 'tab:orange', 'tab:grey']
    ax1.set_xlabel(x_label, fontsize=21)
    ax1.set_ylabel(fairness_label, fontsize=21)
        
    labels = ["ERM", "ERM-Welfare", "ERM-Group Envy Free", "ERM-Group Equitable"]
    #labels = ["ERM", "ERM-Group Envy Free", "ERM-Envy Free"]
    #labels = ["ERM-Group Envy Free", "ERM-Group Equitable", "ERM-Envy Free"]
    for i in range(num_files):
        #ax1.plot(f_x_vals[i], f_train[i], color=colors[i], linewidth=4, label=labels[i])
        ax1.plot(f_x_vals[i], f_test[i], color=colors[i], linewidth=4, label=labels[i])
        #ax1.semilogy(f_x_vals[i], f_test[i], color=colors[i], linewidth=4, label=labels[i])
        if plot_var:
            #ax1.fill_between(f_x_vals[i], f_train[i]-f_train_var[i], f_train[i]+f_train_var[i], color=colors[i], alpha=0.35)
            ax1.fill_between(f_x_vals[i], f_test[i]-f_test_var[i], f_test[i]+f_test_var[i], color=colors[i], alpha=0.35)

    ax1.tick_params(axis='x', labelsize=17)
    ax1.tick_params(axis='y', labelsize=17)
    #plt.xticks([25,50,75,100,125,150])
    #plt.xticks([30,40,50,60,70,80]) 
    #plt.xticks([2,3,4,5,6,7,8,9]) 
    #plt.yticks([0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21])
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(plt_name)

if __name__ == "__main__":
    """ Run this file as "./plot_from_csv <csv_file_names> <metric> <plot_label> <x_label> <plot_var true/false
        Pass in files in this order: erm, erm_welfare, erm_envy, erm_equity
    """
    
    metrics_dict = {'Avg. Loss':[1,2,3,4], 'Welfare':[5,6,7,8], 'Avg. Group Envy':[9,10,11,12], \
            'Group Envy Vio':[13,14,15,16], 'Avg. Group Inequity':[17,18,19,20], 'Avg. Envy':[25,26,27,28], 
            'Individual Envy Vio':[29,30,31,32], "Time (s)":[33,34,33,34]}
    inp_files = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    files = []
    for inp_file in inp_files:
        if inp_file is not "_":
            files.append(inp_file)

    metric = sys.argv[5]
    assert(metric in metrics_dict.keys())
    plot_label = sys.argv[6]
    x_label = sys.argv[7]
    plot_var = sys.argv[8].lower() == 'true'
   
    indicies = metrics_dict[metric]
    data = read_from_csv(files, indicies)
    labels = (plot_label, x_label, metric)
    plot(labels, data, plot_var=plot_var)

