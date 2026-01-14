#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Author Tim Maniquet
Created 2026/01/14 16:48:43
'''

import pickle
import numpy as np

def read_pickle_dict(file_dir):
    '''Read a pickle file as a dictionary.'''
    with open(file_dir, 'rb') as file:
        pickle_dict = pickle.load(file)
    
    return pickle_dict

def remove_diag_flatten(matrix):
    '''
    Remove the diagonal of a matrix and return it flattened.
    '''
    return matrix[~(np.identity(matrix.shape[0], dtype=bool))].ravel()

def upper_triangle(matrix):
    '''
    Return the flattened upper triangle of a matrix
    '''
    return matrix[np.triu_indices(matrix.shape[0], k=1)]

def progress_bar(n_iterations, current_iteration, n_chunks=5):
    '''
    Give some progress on the advancement of a for loop by
    printing a progress bar-looking string.
    '''
    if current_iteration == 0:
        portion = (current_iteration) / (n_iterations // n_chunks)
        progress_bar = '*' * int(portion) + '_' * (n_chunks - int(portion))
        print(f'Progress: [{progress_bar}] {current_iteration + 1}/{n_iterations} iteration complete')

    elif (current_iteration + 1) % (n_iterations // n_chunks) == 0:
        portion = (current_iteration + 1) / (n_iterations // n_chunks)
        progress_bar = '*' * int(portion) + '_' * (n_chunks - int(portion))
        print(f'Progress: [{progress_bar}] {current_iteration + 1}/{n_iterations} iterations complete')
    
        if (current_iteration + 1) == n_iterations:
            print(f'All {n_iterations} iterations completed')

def p_to_star(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"