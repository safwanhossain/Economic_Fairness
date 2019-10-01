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
            # 80% confidence - scale of 1.28
            scale = 1.28
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

def plot(labels, data, file_names, plot_var=False):
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
    ax1.set_xlabel(x_label, fontsize=21)
    ax1.set_ylabel(fairness_label, fontsize=21)
        
    for i in range(num_files):
        ax1.plot(f_x_vals[i], f_train[i], color=colors[i], linewidth=4, label=file_names[i])
        ax1.plot(f_x_vals[i], f_test[i], color=colors[i], linestyle=":", linewidth=4)
        if plot_var:
            ax1.fill_between(f_x_vals[i], f_train[i]-f_train_var[i], f_train[i]+f_train_var[i], color=colors[i], alpha=0.35)
            ax1.fill_between(f_x_vals[i], f_test[i]-f_test_var[i], f_test[i]+f_test_var[i], color=colors[i], alpha=0.35)

    ax1.tick_params(axis='x', labelsize=17)
    ax1.tick_params(axis='y', labelsize=17)
    
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(plt_name)

if __name__ == "__main__":
    """ Run this file as "./plot_from_csv <csv_file_names> <metric> <plot_label> <x_label> <plot_var true/false"""
    metrics_dict = {'loss':[1,2,3,4], 'welfare':[5,6,7,8], 'envy':[9,10,11,12], 'equi':[17,18,19,20]}
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
    plot(labels, data, files, plot_var=plot_var)

