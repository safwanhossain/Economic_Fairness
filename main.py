#!/usr/bin/python3
import numpy as np
import sys, csv
import time

from sklearn.preprocessing import normalize
from constants import *
from helpers import *
from erm_gef_training import train_erm_gef
from erm_equi_training import train_erm_equi
from erm_welfare_training import train_erm_welfare
from erm_training import train_erm
from erm_envy_free_training import train_erm_envy_free

# For our experiments, we will be sweeping ns, d, and lambda values over a range
# Our goal is to analyze their effect on the training and test loss, and the training and test fairness
    # For each experiments, a random L and U matrix will be chosen; then a number of simulation
    # using different train and test sets will be run, and their average taken as the data point

def run_simulation(tup):
    ns_, lambda_val, group_dist, func, L_mat, U_mat, train_Xs, test_Xs, norm = tup
   
    beta_values = 0
    i = 0
    while(beta_values == 0):
        if i == len(train_Xs):
            print("Exhausted all x values")
            return (None,None)

        train_X, test_X = train_Xs[i], test_Xs[i]
        train_s_by_group = define_groups(train_X, group_dist, norm)
        test_s_by_group = define_groups(test_X, group_dist, norm)
        if norm:
            train_X = normalize(train_X, axis=1, norm='l1')    
            test_X = normalize(test_X, axis=1, norm='l1')    
        
        start_time = time.time()
        beta_values, learned_predictions, learned_pred_group, opt_alphas =\
                func(train_X, L_mat, U_mat, train_s_by_group, lamb=lambda_val)
        end_time = time.time()
        total_time = end_time - start_time
        if beta_values == 0:
            print("EXCEPTION WAS RAISED")
        i += 1
    
    # Get all the data for training data
    tr_total_loss = compute_final_loss(opt_alphas, L_mat, train_X, learned_predictions)
    tr_total_welfare = compute_welfare(opt_alphas, U_mat, train_X, learned_predictions)
    tr_total_envy, tr_envy_violations = total_group_envy(opt_alphas, U_mat, train_s_by_group, \
            learned_pred_group) 
    tr_total_equi, tr_equi_violations = total_group_equi(opt_alphas, U_mat, train_s_by_group, \
            learned_pred_group) 
    tr_avg_envy, tr_in_envy_violations = total_average_envy(opt_alphas, U_mat, train_X, \
            learned_predictions)
    train_data = (tr_total_loss, tr_total_welfare, tr_total_envy, tr_envy_violations, \
            tr_total_equi, tr_equi_violations, tr_avg_envy, tr_in_envy_violations, total_time)
    
    # Get all the data for test data
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(beta_values, test_X, test_s_by_group, K)
    st_total_loss = compute_final_loss(opt_alphas, L_mat, test_X, st_learned_predictions)
    st_total_welfare = compute_welfare(opt_alphas, U_mat, test_X, st_learned_predictions)
    st_total_envy, st_envy_violations = total_group_envy(opt_alphas, U_mat, test_s_by_group, \
            st_learned_pred_group) 
    st_total_equi, st_equi_violations = total_group_equi(opt_alphas, U_mat, test_s_by_group, \
            st_learned_pred_group) 
    st_avg_envy, st_in_envy_violations = total_average_envy(opt_alphas, U_mat, test_X, \
            st_learned_predictions)
    test_data = (st_total_loss, st_total_welfare, st_total_envy, st_envy_violations, \
            st_total_equi, st_equi_violations, st_avg_envy, st_in_envy_violations)
    
    return (train_data, test_data)

def sweep_ns_parameters_parallel(ns_vals, func, lambda_val, group_dist, L_mats, U_mats):
    # TODO: Fix how fairness is stored? 
    # TODO: Pass functions and not strings for distribution parameters
    """ fairness_func is the function used to compute fairness; None for ERM_loss
        lu_mat is a tuple of the loss and utility matrix 
        group_dist is a list, with each element representing the propotion of individuals in that group
            If groups are not required (for ERM and ERM_welfare), pass in NONE
        sample_dist is a dictionary with keys: ['data', 'loss', 'utility']. It gives the distribution
            to use for the generation of each. If None, Uniform(0,1) will be used for all
    """
    import concurrent.futures 
    
    filename = "sweep_n_100sim_"+func.__name__+".csv"
    print("Params: lambda=", lambda_val, "num_sim=", num_sims) 
    print("Saving results to", filename)
    
    csv_file = open(filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    num_vals = len(ns_vals)

    # These store the average values for each ns
    train_losses, test_losses, train_losses_var, test_losses_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_welfare, test_welfare, train_welfare_var, test_welfare_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_envy, test_envy, train_envy_var, test_envy_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_envy_vio, test_envy_vio, train_envy_vio_var, test_envy_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_equi, test_equi, train_equi_var, test_equi_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_equi_vio, test_equi_vio, train_equi_vio_var, test_equi_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_in_envy, test_in_envy, train_in_envy_var, test_in_envy_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_in_envy_vio, test_in_envy_vio, train_in_envy_vio_var, test_in_envy_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_time, train_time_var = np.ones(num_vals), np.ones(num_vals)
    
    for index, ns_ in enumerate(ns_vals):
        ns_ = int(ns_)
        print("ns: ", ns_)
        
        # these store values for every simulation - will be averaged
        c_train_losses, c_test_losses = np.ones(num_sims), np.ones(num_sims)
        c_train_welfare, c_test_welfare = np.ones(num_sims), np.ones(num_sims)
        c_train_envy, c_test_envy = np.ones(num_sims), np.ones(num_sims)
        c_train_envy_vio, c_test_envy_vio = np.ones(num_sims), np.ones(num_sims)
        c_train_equi, c_test_equi = np.ones(num_sims), np.ones(num_sims)
        c_train_equi_vio, c_test_equi_vio = np.ones(num_sims), np.ones(num_sims)
        c_train_in_envy, c_test_in_envy = np.ones(num_sims), np.ones(num_sims)
        c_train_in_envy_vio, c_test_in_envy_vio = np.ones(num_sims), np.ones(num_sims)
        c_train_time = np.ones(num_sims)

        inputs = []
        for sim in range(num_sims):
            L_mat = L_mats[sim]
            U_mat = U_mats[sim]
            train_Xs = [generate_data(ns_, m, 'uniform') for i in range(10)]
            test_Xs = [generate_data(nt, m, 'uniform') for i in range(10)]
            
            tup = (ns_, lambda_val, group_dist, func, L_mat, U_mat, train_Xs, test_Xs, False)
            inputs.append(tup)

        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)
        
        for sim, future in enumerate(futures):
            assert(future.done())
            train_data, test_data = future.result()
            if train_data == None:
                c_train_losses[sim], c_train_welfare[sim], c_train_envy[sim], c_train_envy_vio[sim], \
                    c_train_equi[sim], c_train_equi_vio[sim], c_train_in_envy[sim], c_train_in_envy_vio[sim] = -1, -1, -1, -1, -1, -1, -1, -1
                c_test_losses[sim], c_test_welfare[sim], c_test_envy[sim], c_test_envy_vio[sim], \
                    c_test_equi[sim], c_test_equi_vio[sim], c_test_in_envy[sim], c_test_in_envy_vio[sim] = -1, -1, -1, -1, -1, -1, -1, -1
            else:
                c_train_losses[sim], c_train_welfare[sim], c_train_envy[sim], c_train_envy_vio[sim], \
                    c_train_equi[sim], c_train_equi_vio[sim], c_train_in_envy[sim], c_train_in_envy_vio[sim], c_train_time[sim] = train_data
                c_test_losses[sim], c_test_welfare[sim], c_test_envy[sim], c_test_envy_vio[sim], \
                    c_test_equi[sim], c_test_equi_vio[sim], c_test_in_envy[sim], c_test_in_envy_vio[sim] = test_data
            
            #print(ns, "Train Equi: ", c_train_equi[sim])
            #print(ns, "Test Equi: ", c_test_equi[sim])
        
        np.delete(c_train_losses, np.where(c_train_losses==-1))
        np.delete(c_test_losses, np.where(c_test_losses==-1))
        np.delete(c_train_welfare, np.where(c_train_welfare==-1))
        np.delete(c_test_welfare, np.where(c_test_welfare==-1))
        np.delete(c_train_envy, np.where(c_train_envy==-1))
        np.delete(c_test_envy, np.where(c_test_envy==-1))
        np.delete(c_train_envy_vio, np.where(c_train_envy_vio==-1))
        np.delete(c_test_envy_vio, np.where(c_test_envy_vio==-1))
        np.delete(c_train_equi, np.where(c_train_equi==-1))
        np.delete(c_test_equi, np.where(c_test_equi==-1))
        np.delete(c_train_equi_vio, np.where(c_train_equi_vio==-1))
        np.delete(c_test_equi_vio, np.where(c_test_equi_vio==-1))
        np.delete(c_train_in_envy, np.where(c_train_in_envy==-1))
        np.delete(c_test_in_envy, np.where(c_test_in_envy==-1))
        np.delete(c_train_in_envy_vio, np.where(c_train_in_envy_vio==-1))
        np.delete(c_test_in_envy_vio, np.where(c_test_in_envy_vio==-1))
        np.delete(c_train_time, np.where(c_train_time==-1))

        train_losses[index], train_losses_var[index] = np.mean(c_train_losses), np.sqrt(np.var(c_train_losses))
        test_losses[index], test_losses_var[index] = np.mean(c_test_losses), np.sqrt(np.var(c_test_losses))
        
        train_welfare[index], train_welfare_var[index] = np.mean(c_train_welfare), np.sqrt(np.var(c_train_welfare))
        test_welfare[index], test_welfare_var[index] = np.mean(c_test_welfare), np.sqrt(np.var(c_test_welfare))
        
        train_envy[index], train_envy_var[index] = np.mean(c_train_envy), np.sqrt(np.var(c_train_envy))
        test_envy[index], test_envy_var[index] = np.mean(c_test_envy), np.sqrt(np.var(c_test_envy))
        
        train_envy_vio[index], train_envy_vio_var[index] = np.mean(c_train_envy_vio), np.sqrt(np.var(c_train_envy_vio))
        test_envy_vio[index], test_envy_vio_var[index] = np.mean(c_test_envy_vio), np.sqrt(np.var(c_test_envy_vio))
        
        train_equi[index], train_equi_var[index] = np.mean(c_train_equi), np.sqrt(np.var(c_train_equi))
        test_equi[index], test_equi_var[index] = np.mean(c_test_equi), np.sqrt(np.var(c_test_equi))
        
        train_equi_vio[index], train_equi_vio_var[index] = np.mean(c_train_equi_vio), np.sqrt(np.var(c_train_equi_vio))
        test_equi_vio[index], test_equi_vio_var[index] = np.mean(c_test_equi_vio), np.sqrt(np.var(c_test_equi_vio))
       
        train_in_envy[index], train_in_envy_var[index] = np.mean(c_train_in_envy), np.sqrt(np.var(c_train_in_envy))
        test_in_envy[index], test_in_envy_var[index] = np.mean(c_test_in_envy), np.sqrt(np.var(c_test_in_envy))
        
        train_in_envy_vio[index], train_in_envy_vio_var[index] = np.mean(c_train_in_envy_vio), np.sqrt(np.var(c_train_in_envy_vio))
        test_in_envy_vio[index], test_in_envy_vio_var[index] = np.mean(c_test_in_envy_vio), np.sqrt(np.var(c_test_in_envy_vio))
        
        train_time[index], train_time_var[index] = np.mean(c_train_time), np.sqrt(np.var(c_train_time))
        
        #print("Envy train and test: ", train_envy[index], test_envy[index])
        #print("Envy vio train and test: ", train_envy_vio[index], test_envy_vio[index])

        row = [str(ns_)] + [str(train_losses[index])] + [str(train_losses_var[index])] +\
                           [str(test_losses[index])] + [str(test_losses_var[index])] +\
                           [str(train_welfare[index])] + [str(train_welfare_var[index])] +\
                           [str(test_welfare[index])] +  [str(test_welfare_var[index])] + \
                           [str(train_envy[index])] +  [str(train_envy_var[index])] + \
                           [str(test_envy[index])] +  [str(test_envy_var[index])] + \
                           [str(train_envy_vio[index])] +  [str(train_envy_vio_var[index])] + \
                           [str(test_envy_vio[index])] +  [str(test_envy_vio_var[index])] + \
                           [str(train_equi[index])] +  [str(train_equi_var[index])] + \
                           [str(test_equi[index])] +  [str(test_equi_var[index])] + \
                           [str(train_equi_vio[index])] +  [str(train_equi_vio_var[index])] + \
                           [str(test_equi_vio[index])] +  [str(test_equi_vio_var[index])] + \
                           [str(train_in_envy[index])] + [str(train_in_envy_var[index])] +\
                           [str(test_in_envy[index])] +  [str(test_in_envy_var[index])] + \
                           [str(train_in_envy_vio[index])] + [str(train_in_envy_vio_var[index])] +\
                           [str(test_in_envy_vio[index])] +  [str(test_in_envy_vio_var[index])] +\
                           [str(train_time[index])] +  [str(train_time_var[index])]

        csv_writer.writerow(row)
        csv_file.flush()

def sweep_g_parameters_parallel(g_vals, func, lambda_val, L_mats, U_mats):
    # TODO: Fix how fairness is stored? 
    # TODO: Pass functions and not strings for distribution parameters
    """ fairness_func is the function used to compute fairness; None for ERM_loss
        lu_mat is a tuple of the loss and utility matrix 
        group_dist is a list, with each element representing the propotion of individuals in that group
            If groups are not required (for ERM and ERM_welfare), pass in NONE
        sample_dist is a dictionary with keys: ['data', 'loss', 'utility']. It gives the distribution
            to use for the generation of each. If None, Uniform(0,1) will be used for all
    """
    import concurrent.futures 
    
    filename = "sweep_g_"+func.__name__+".csv"
    print("Params: lambda=", lambda_val, "num_sim=", num_sims) 
    print("Saving results to", filename)
    
    csv_file = open(filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    num_vals = len(g_vals)

    # These store the average values for each ns
    train_losses, test_losses, train_losses_var, test_losses_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_welfare, test_welfare, train_welfare_var, test_welfare_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_envy, test_envy, train_envy_var, test_envy_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_envy_vio, test_envy_vio, train_envy_vio_var, test_envy_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_equi, test_equi, train_equi_var, test_equi_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_equi_vio, test_equi_vio, train_equi_vio_var, test_equi_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_in_envy, test_in_envy, train_in_envy_var, test_in_envy_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_in_envy_vio, test_in_envy_vio, train_in_envy_vio_var, test_in_envy_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_time, train_time_var = np.ones(num_vals), np.ones(num_vals)
    
    for index, g_ in enumerate(g_vals):
        g_ = int(g_)
        print("ns:", ns, "g:", g_)
        group_dist = [1/g_ for i in range(g_)]
        print(group_dist)
        # these store values for every simulation - will be averaged
        c_train_losses, c_test_losses = np.ones(num_sims), np.ones(num_sims)
        c_train_welfare, c_test_welfare = np.ones(num_sims), np.ones(num_sims)
        c_train_envy, c_test_envy = np.ones(num_sims), np.ones(num_sims)
        c_train_envy_vio, c_test_envy_vio = np.ones(num_sims), np.ones(num_sims)
        c_train_equi, c_test_equi = np.ones(num_sims), np.ones(num_sims)
        c_train_equi_vio, c_test_equi_vio = np.ones(num_sims), np.ones(num_sims)
        c_train_in_envy, c_test_in_envy = np.ones(num_sims), np.ones(num_sims)
        c_train_in_envy_vio, c_test_in_envy_vio = np.ones(num_sims), np.ones(num_sims)
        c_train_time = np.ones(num_sims)

        inputs = []
        for sim in range(num_sims):
            L_mat = L_mats[sim]
            U_mat = U_mats[sim]
            train_Xs = [generate_data(ns, m, 'uniform') for i in range(10)]
            test_Xs = [generate_data(nt, m, 'uniform') for i in range(10)]
        
            tup = (ns, lambda_val, group_dist, func, L_mat, U_mat, train_Xs, test_Xs, False)
            inputs.append(tup)

        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)
        
        for sim, future in enumerate(futures):
            assert(future.done())
            train_data, test_data = future.result()
            if train_data == None:
                c_train_losses[sim], c_train_welfare[sim], c_train_envy[sim], c_train_envy_vio[sim], \
                    c_train_equi[sim], c_train_equi_vio[sim], c_train_in_envy[sim], c_train_in_envy_vio[sim] = -1, -1, -1, -1, -1, -1, -1, -1
                c_test_losses[sim], c_test_welfare[sim], c_test_envy[sim], c_test_envy_vio[sim], \
                    c_test_equi[sim], c_test_equi_vio[sim], c_test_in_envy[sim], c_test_in_envy_vio[sim] = -1, -1, -1, -1, -1, -1, -1, -1
            else:
                c_train_losses[sim], c_train_welfare[sim], c_train_envy[sim], c_train_envy_vio[sim], \
                    c_train_equi[sim], c_train_equi_vio[sim], c_train_in_envy[sim], c_train_in_envy_vio[sim], c_train_time[sim] = train_data
                c_test_losses[sim], c_test_welfare[sim], c_test_envy[sim], c_test_envy_vio[sim], \
                    c_test_equi[sim], c_test_equi_vio[sim], c_test_in_envy[sim], c_test_in_envy_vio[sim] = test_data

        np.delete(c_train_losses, np.where(c_train_losses==-1))
        np.delete(c_test_losses, np.where(c_test_losses==-1))
        np.delete(c_train_welfare, np.where(c_train_welfare==-1))
        np.delete(c_test_welfare, np.where(c_test_welfare==-1))
        np.delete(c_train_envy, np.where(c_train_envy==-1))
        np.delete(c_test_envy, np.where(c_test_envy==-1))
        np.delete(c_train_envy_vio, np.where(c_train_envy_vio==-1))
        np.delete(c_test_envy_vio, np.where(c_test_envy_vio==-1))
        np.delete(c_train_equi, np.where(c_train_equi==-1))
        np.delete(c_test_equi, np.where(c_test_equi==-1))
        np.delete(c_train_equi_vio, np.where(c_train_equi_vio==-1))
        np.delete(c_test_equi_vio, np.where(c_test_equi_vio==-1))
        np.delete(c_train_in_envy_vio, np.where(c_train_in_envy_vio==-1))
        np.delete(c_test_in_envy_vio, np.where(c_test_in_envy_vio==-1))
        np.delete(c_train_time, np.where(c_train_time==-1))

        train_losses[index], train_losses_var[index] = np.mean(c_train_losses), np.sqrt(np.var(c_train_losses))
        test_losses[index], test_losses_var[index] = np.mean(c_test_losses), np.sqrt(np.var(c_test_losses))
        
        train_welfare[index], train_welfare_var[index] = np.mean(c_train_welfare), np.sqrt(np.var(c_train_welfare))
        test_welfare[index], test_welfare_var[index] = np.mean(c_test_welfare), np.sqrt(np.var(c_test_welfare))
        
        train_envy[index], train_envy_var[index] = np.mean(c_train_envy), np.sqrt(np.var(c_train_envy))
        test_envy[index], test_envy_var[index] = np.mean(c_test_envy), np.sqrt(np.var(c_test_envy))
        
        train_envy_vio[index], train_envy_vio_var[index] = np.mean(c_train_envy_vio), np.sqrt(np.var(c_train_envy_vio))
        test_envy_vio[index], test_envy_vio_var[index] = np.mean(c_test_envy_vio), np.sqrt(np.var(c_test_envy_vio))
        
        train_equi[index], train_equi_var[index] = np.mean(c_train_equi), np.sqrt(np.var(c_train_equi))
        test_equi[index], test_equi_var[index] = np.mean(c_test_equi), np.sqrt(np.var(c_test_equi))
        
        train_equi_vio[index], train_equi_vio_var[index] = np.mean(c_train_equi_vio), np.sqrt(np.var(c_train_equi_vio))
        test_equi_vio[index], test_equi_vio_var[index] = np.mean(c_test_equi_vio), np.sqrt(np.var(c_test_equi_vio))
        
        train_in_envy[index], train_in_envy_var[index] = np.mean(c_train_in_envy), np.sqrt(np.var(c_train_in_envy))
        test_in_envy[index], test_in_envy_var[index] = np.mean(c_test_in_envy), np.sqrt(np.var(c_test_in_envy))
        
        train_in_envy_vio[index], train_in_envy_vio_var[index] = np.mean(c_train_in_envy_vio), np.sqrt(np.var(c_train_in_envy_vio))
        test_in_envy_vio[index], test_in_envy_vio_var[index] = np.mean(c_test_in_envy_vio), np.sqrt(np.var(c_test_in_envy_vio))
       
        train_time[index], train_time_var[index] = np.mean(c_train_time), np.sqrt(np.var(c_train_time))
        
        #print("Envy train and test: ", train_envy[index], test_envy[index])
        #print("Envy vio train and test: ", train_envy_vio[index], test_envy_vio[index])

        row = [str(g_)] + [str(train_losses[index])] + [str(train_losses_var[index])] +\
                           [str(test_losses[index])] + [str(test_losses_var[index])] +\
                           [str(train_welfare[index])] + [str(train_welfare_var[index])] +\
                           [str(test_welfare[index])] +  [str(test_welfare_var[index])] + \
                           [str(train_envy[index])] +  [str(train_envy_var[index])] + \
                           [str(test_envy[index])] +  [str(test_envy_var[index])] + \
                           [str(train_envy_vio[index])] +  [str(train_envy_vio_var[index])] + \
                           [str(test_envy_vio[index])] +  [str(test_envy_vio_var[index])] + \
                           [str(train_equi[index])] +  [str(train_equi_var[index])] + \
                           [str(test_equi[index])] +  [str(test_equi_var[index])] + \
                           [str(train_equi_vio[index])] +  [str(train_equi_vio_var[index])] + \
                           [str(test_equi_vio[index])] +  [str(test_equi_vio_var[index])] +\
                           [str(train_in_envy_vio[index])] + [str(train_in_envy_vio_var[index])] +\
                           [str(test_in_envy_vio[index])] +  [str(test_in_envy_vio_var[index])] +\
                           [str(train_time[index])] +  [str(train_time_var[index])]

        csv_writer.writerow(row)
        csv_file.flush()

def envy_experiment(func, lambda_val, L_mats, U_mats_dict):
    # TODO: Fix how fairness is stored? 
    # TODO: Pass functions and not strings for distribution parameters
    """ fairness_func is the function used to compute fairness; None for ERM_loss
        lu_mat is a tuple of the loss and utility matrix 
        group_dist is a list, with each element representing the propotion of individuals in that group
            If groups are not required (for ERM and ERM_welfare), pass in NONE
        sample_dist is a dictionary with keys: ['data', 'loss', 'utility']. It gives the distribution
            to use for the generation of each. If None, Uniform(0,1) will be used for all
    """
    import concurrent.futures 
    
    filename = "envy_experiment_"+func.__name__+".csv"
    print("Params: lambda=", lambda_val, "num_sim=", num_sims) 
    print("Saving results to", filename)
    var_vals = U_mats_dict.keys()
    print(var_vals)

    csv_file = open(filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    num_vals = len(var_vals)

    # These store the average values for each ns
    train_losses, test_losses, train_losses_var, test_losses_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_envy, test_envy, train_envy_var, test_envy_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_envy_vio, test_envy_vio, train_envy_vio_var, test_envy_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_in_envy, test_in_envy, train_in_envy_var, test_in_envy_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    train_in_envy_vio, test_in_envy_vio, train_in_envy_vio_var, test_in_envy_vio_var = \
            np.ones(num_vals), np.ones(num_vals), np.ones(num_vals), np.ones(num_vals)
    
    for index, var_ in enumerate(var_vals):
        ns = 50
        print("Var: ", var_, "ns: ", ns)
        group_dist = [0.25, 0.25, 0.25, 0.25]

        # these store values for every simulation - will be averaged
        c_train_losses, c_test_losses = np.ones(num_sims), np.ones(num_sims)
        c_train_envy, c_test_envy = np.ones(num_sims), np.ones(num_sims)
        c_train_envy_vio, c_test_envy_vio = np.ones(num_sims), np.ones(num_sims)
        c_train_in_envy, c_test_in_envy = np.ones(num_sims), np.ones(num_sims)
        c_train_in_envy_vio, c_test_in_envy_vio = np.ones(num_sims), np.ones(num_sims)

        inputs = []
        for sim in range(num_sims):
            L_mat = L_mats[sim]
            U_mat = U_mats_dict[var_][sim]
            train_Xs = [generate_data(ns, m, 'uniform') for i in range(10)]
            test_Xs = [generate_data(nt, m, 'uniform') for i in range(10)]
        
            tup = (ns, lambda_val, group_dist, func, L_mat, U_mat, train_Xs, test_Xs, True)
            inputs.append(tup)

        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)
        
        for sim, future in enumerate(futures):
            assert(future.done())
            train_data, test_data = future.result()
            if train_data == None:
                c_train_losses[sim], c_train_envy[sim], c_train_envy_vio[sim],\
                    c_train_in_envy[sim], c_train_in_envy_vio[sim] = -1, -1, -1, -1, -1
                c_test_losses[sim], c_test_envy[sim], c_test_envy_vio[sim], \
                        c_test_in_envy[sim], c_test_in_envy_vio[sim] = -1, -1, -1, -1, -1
            else:
                c_train_losses[sim], _, c_train_envy[sim], c_train_envy_vio[sim], _, _, \
                        c_train_in_envy[sim], c_train_in_envy_vio[sim] = train_data
                c_test_losses[sim], _, c_test_envy[sim], c_test_envy_vio[sim], _, _, \
                        c_test_in_envy[sim], c_test_in_envy_vio[sim] = test_data

        np.delete(c_train_losses, np.where(c_train_losses==-1))
        np.delete(c_test_losses, np.where(c_test_losses==-1))
        np.delete(c_train_envy, np.where(c_train_envy==-1))
        np.delete(c_test_envy, np.where(c_test_envy==-1))
        np.delete(c_train_envy_vio, np.where(c_train_envy_vio==-1))
        np.delete(c_test_envy_vio, np.where(c_test_envy_vio==-1))
        np.delete(c_train_in_envy, np.where(c_train_in_envy==-1))
        np.delete(c_test_in_envy, np.where(c_test_in_envy==-1))
        np.delete(c_train_in_envy_vio, np.where(c_train_in_envy_vio==-1))
        np.delete(c_test_in_envy_vio, np.where(c_test_in_envy_vio==-1))

        train_losses[index], train_losses_var[index] = np.mean(c_train_losses), np.sqrt(np.var(c_train_losses))
        test_losses[index], test_losses_var[index] = np.mean(c_test_losses), np.sqrt(np.var(c_test_losses))
        
        train_envy[index], train_envy_var[index] = np.mean(c_train_envy), np.sqrt(np.var(c_train_envy))
        test_envy[index], test_envy_var[index] = np.mean(c_test_envy), np.sqrt(np.var(c_test_envy))
        
        train_envy_vio[index], train_envy_vio_var[index] = np.mean(c_train_envy_vio), np.sqrt(np.var(c_train_envy_vio))
        test_envy_vio[index], test_envy_vio_var[index] = np.mean(c_test_envy_vio), np.sqrt(np.var(c_test_envy_vio))
        
        train_in_envy[index], train_in_envy_var[index] = np.mean(c_train_in_envy), np.sqrt(np.var(c_train_in_envy))
        test_in_envy[index], test_in_envy_var[index] = np.mean(c_test_in_envy), np.sqrt(np.var(c_test_in_envy))
        
        train_in_envy_vio[index], train_in_envy_vio_var[index] = np.mean(c_train_in_envy_vio), np.sqrt(np.var(c_train_in_envy_vio))
        test_in_envy_vio[index], test_in_envy_vio_var[index] = np.mean(c_test_in_envy_vio), np.sqrt(np.var(c_test_in_envy_vio))
        
        row = [str(var_)] + [str(train_losses[index])] + [str(train_losses_var[index])] +\
                           [str(test_losses[index])] + [str(test_losses_var[index])] +\
                           [str(0)] + [str(0)] +\
                           [str(0)] +  [str(0)] + \
                           [str(train_envy[index])] +  [str(train_envy_var[index])] + \
                           [str(test_envy[index])] +  [str(test_envy_var[index])] + \
                           [str(train_envy_vio[index])] +  [str(train_envy_vio_var[index])] + \
                           [str(test_envy_vio[index])] +  [str(test_envy_vio_var[index])] + \
                           [str(0)] +  [str(0)] + \
                           [str(0)] +  [str(0)] + \
                           [str(0)] +  [str(0)] + \
                           [str(0)] +  [str(0)] +\
                           [str(train_in_envy[index])] + [str(train_in_envy_var[index])] +\
                           [str(test_in_envy[index])] +  [str(test_in_envy_var[index])] + \
                           [str(train_in_envy_vio[index])] + [str(train_in_envy_vio_var[index])] +\
                           [str(test_in_envy_vio[index])] +  [str(test_in_envy_vio_var[index])]

        csv_writer.writerow(row)
        csv_file.flush()

def benchmark_erm():
    # Following values were used for this benchmark test: lambda_welf=1, num_sims=50, NUM_CORES=16
    # It took 57.5s on my Vector Workstation (not server)
    ns_vals = [30, 40]
    group_dist = [0.25, 0.25,0.25,0.25]
    print("Here: ", sum(group_dist))
    start = time.time()
    sweep_ns_parameters_parallel(ns_vals, train_erm, 5, group_dist)
    end = time.time()
    print("ERM took: ", end-start)

def benchmark_erm_welfare():
    # Following values were used for this benchmark test: lambda_welf=1, num_sims=50, NUM_CORES=16
    # It took 34.2s on my Vector Workstation (not server)
    ns_vals = [30, 40]
    group_dist = [0.25, 0.25, 0.25, 0.25]
    
    start = time.time()
    sweep_ns_parameters_parallel(ns_vals, train_erm_welfare, 5, group_dist)
    end = time.time()
    print("ERM welfare took: ", end-start)

def benchmark_erm_envy():
    # Following values were used for this benchmark test: lambda_equi=1, num_sims=50, NUM_CORES=16
    # It took 126.708s on my Vector Workstation (not server)
    ns_vals = [30, 40]
    group_dist = [0.25, 0.25, 0.25, 0.25]
    
    start = time.time()
    sweep_ns_parameters_parallel(ns_vals, train_erm_gef, 10, group_dist)
    end = time.time()
    print("ERM envy free took: ", end-start)

def run_envy_experiment(L_mats, typ):
    all_vars = [0, 0.2, 0.4, 0.6, 0.8]
    U_mat_dict = {}
    for curr_var in all_vars:
        name = "U_mat_" + str(curr_var) + "_file"
        #U_mats = [generate_utility_matrix_var(d, m, 'uniform', curr_var) for i in range(num_sims)]
        #np.savez(name, *U_mats)
        u_file = np.load(name+'.npz')
        U_mats = [u_file[i] for i in u_file.files]
        U_mat_dict[curr_var] = U_mats

    if typ == "in_envy":
        envy_experiment(train_erm_envy_free, 10, L_mats, U_mat_dict)
    elif typ == "g_envy":
        envy_experiment(train_erm_gef, 10, L_mats, U_mat_dict)
    else:
        envy_experiment(train_erm, 10, L_mats, U_mat_dict)
    
def ns_experiment(L_mats, U_mats):
    ns_vals = [65, 70, 75, 80]
    group_dist = [0.25, 0.25, 0.25, 0.25]
    
    #start = time.time()
    #sweep_ns_parameters_parallel(ns_vals, train_erm, 0, group_dist, L_mats, U_mats)
    #end = time.time()
    #print("ERM experiment took:", end-start)
    
    #start = time.time()
    #sweep_ns_parameters_parallel(ns_vals, train_erm_welfare, 10, group_dist, L_mats, U_mats)
    #end = time.time()
    #print("ERM welfare took: ", end-start)
    
    #start = time.time()
    #sweep_ns_parameters_parallel(ns_vals, train_erm_equi, 10, group_dist, L_mats, U_mats)
    #end = time.time()
    #print("ERM equi took: ", end-start)
    
    #start = time.time()
    #sweep_ns_parameters_parallel(ns_vals, train_erm_gef, 10, group_dist, L_mats, U_mats)
    #end = time.time()
    #print("ERM group envy free took: ", end-start)
    
    start = time.time()
    sweep_ns_parameters_parallel(ns_vals, train_erm_envy_free, 10, group_dist, L_mats, U_mats)
    end = time.time()
    print("ERM envy free took: ", end-start)
    
    
def g_experiment(L_mats, U_mats):
    g_vals = [2,3,4,5,6,7,8,9] 
    
    start = time.time()
    sweep_g_parameters_parallel(g_vals, train_erm, 0, L_mats, U_mats)
    end = time.time()
    print("ERM experiment took:", end-start)
    
    start = time.time()
    sweep_g_parameters_parallel(g_vals, train_erm_welfare, 10, L_mats, U_mats)
    end = time.time()
    print("ERM welfare took: ", end-start)
    
    start = time.time()
    sweep_g_parameters_parallel(g_vals, train_erm_equi, 10, L_mats, U_mats)
    end = time.time()
    print("ERM equi took: ", end-start)
    
    g_vals = [5,6,7,8,9] 
    start = time.time()
    sweep_g_parameters_parallel(g_vals, train_erm_gef, 10, L_mats, U_mats)
    end = time.time()
    print("ERM envy free took: ", end-start)

def main():
    #benchmark_erm()
    #benchmark_erm_welfare()
    #benchmark_erm_envy()
        
    #L_mats = [generate_loss_matrix(d, m, 'uniform') for i in range(num_sims)]
    #U_mats = [generate_utility_matrix(d, m, 'uniform') for i in range(num_sims)]
    #np.savez("L_mat_file", *L_mats)
    #np.savez("U_mat_file", *U_mats)

    l_file = np.load('L_mat_file.npz')
    L_mats = [l_file[i] for i in l_file.files]
    u_file = np.load('U_mat_file.npz')
    U_mats = [u_file[i] for i in u_file.files]

    if sys.argv[1] == "ns":
        ns_experiment(L_mats, U_mats)
    if sys.argv[1] == "g":
        g_experiment(L_mats, U_mats)
    if sys.argv[1] == "envy":
        run_envy_experiment(L_mats, "erm")
   
if __name__ == "__main__":
    main()

