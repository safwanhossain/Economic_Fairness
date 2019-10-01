#!/usr/bin/python3
import numpy as np
import sys, csv
import time

from constants import *
from helpers import *
from erm_gef_training import train_erm_gef
from erm_welfare_training import train_erm_welfare
from erm_training import train_erm

# For our experiments, we will be sweeping ns, d, and lambda values over a range
# Our goal is to analyze their effect on the training and test loss, and the training and test fairness
    # For each experiments, a random L and U matrix will be chosen; then a number of simulation
    # using different train and test sets will be run, and their average taken as the data point

def run_simulation(tup):
    ns_, lambda_val, group_dist, func, L_mat, U_mat, train_X, test_X = tup
    
    train_s_by_group = define_groups(train_X, group_dist)
    test_s_by_group = define_groups(test_X, group_dist)

    beta_values, learned_predictions, learned_pred_group, opt_alphas =\
            func(train_X, L_mat, U_mat, train_s_by_group, lamb=lambda_val) 
    
    # Get all the data for training data
    tr_total_loss = compute_final_loss(opt_alphas, L_mat, train_X, learned_predictions)
    tr_total_welfare = compute_welfare(opt_alphas, U_mat, train_X, learned_predictions)
    tr_total_envy, tr_envy_violations = total_group_envy(opt_alphas, U_mat, train_s_by_group, \
            learned_pred_group) 
    tr_total_equi, tr_equi_violations = total_group_equi(opt_alphas, U_mat, train_s_by_group, \
            learned_pred_group) 
    train_data = (tr_total_loss, tr_total_welfare, tr_total_envy, tr_envy_violations, \
            tr_total_equi, tr_equi_violations)
    
    # Get all the data for training data
    st_learned_predictions, st_learned_pred_group = \
            get_all_predictions(beta_values, test_X, test_s_by_group, K)
    st_total_loss = compute_final_loss(opt_alphas, L_mat, test_X, st_learned_predictions)
    st_total_welfare = compute_welfare(opt_alphas, U_mat, test_X, st_learned_predictions)
    st_total_envy, st_envy_violations = total_group_envy(opt_alphas, U_mat, test_s_by_group, \
            st_learned_pred_group) 
    st_total_equi, st_equi_violations = total_group_equi(opt_alphas, U_mat, test_s_by_group, \
            st_learned_pred_group) 
    test_data = (st_total_loss, st_total_welfare, st_total_envy, st_envy_violations, \
            st_total_equi, st_equi_violations)
    
    return (train_data, test_data)

def sweep_ns_parameters_parallel(ns_vals, func, lambda_val, group_dist):
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
    
    filename = "sweep_n_"+func.__name__+".csv"
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

        inputs = []
        for sim in range(num_sims):
            L_mat = generate_loss_matrix(d, m, 'uniform')
            U_mat = generate_utility_matrix(d, m, 'uniform')
            train_X = generate_data(ns_, m, 'uniform')
            test_X = generate_data(nt, m, 'uniform')
            
            tup = (ns_, lambda_val, group_dist, func, L_mat, U_mat, train_X, test_X)
            inputs.append(tup)

        executor = concurrent.futures.ProcessPoolExecutor(NUM_CORES)
        futures = [executor.submit(run_simulation, item) for item in inputs]
        concurrent.futures.wait(futures)
        #futures = [run_simulation(item) for item in inputs]
        for sim, future in enumerate(futures):
            assert(future.done())
            train_data, test_data = future.result()
            #train_data, test_data = future
            c_train_losses[sim], c_train_welfare[sim], c_train_envy[sim], c_train_envy_vio[sim], \
                c_train_equi[sim], c_train_equi_vio[sim] = train_data
            c_test_losses[sim], c_test_welfare[sim], c_test_envy[sim], c_test_envy_vio[sim], \
                c_test_equi[sim], c_test_equi_vio[sim] = test_data

        #print(c_train_losses)
        #print(c_test_losses)
        #print(c_train_welfare)
        #print(c_test_welfare)

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
       
        print("Envy train and test: ", train_envy[index], test_envy[index])
        print("Envy vio train and test: ", train_envy_vio[index], test_envy_vio[index])

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
                           [str(test_equi_vio[index])] +  [str(test_equi_vio_var[index])]

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

def main():
    #ns_vals = [100,120,140,160,180]
    ns_vals = [10, 20, 40, 50, 60, 75, 80, 100, 120]
    group_dist = [0.25, 0.25, 0.25, 0.25]
    
    #benchmark_erm()
    #benchmark_erm_welfare()
    #benchmark_erm_envy()

    run = True
    if run:
        start = time.time()
        sweep_ns_parameters_parallel(ns_vals, train_erm, 0, group_dist)
        end = time.time()
        print("ERM experiment took:", end-start)
        
        start = time.time()
        sweep_ns_parameters_parallel(ns_vals, train_erm_welfare, 5, group_dist)
        end = time.time()
        print("ERM welfare took: ", end-start)
        
        #start = time.time()
        #sweep_ns_parameters_parallel(ns_vals, train_erm_gef, 10, group_dist)
        #end = time.time()
        #print("ERM envy free took: ", end-start)

if __name__ == "__main__":
    main()


