#!/usr/bin/python

from __future__ import division, print_function
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from ast import literal_eval
import argparse
from itertools import product
import time
import os
import sys

# see the accompanying notebook version for more information on the general strategy (how to prioritize the
# matchings) and the linear program (the idea of a 3D tensor, constraints, etc.)
# it also shows the form of intermediate data (e.g. the data frames of tutor/tutee info) which may be useful to see


def load_tutor_info(data_path, verbose=False):
    """
    Reads in the tutor data which must be in a file named tutor_info.txt in the directory specified by data_path.
    Columns expected in the tutor_info file are tutor id, tutor name, list of classes they will tutor, number of
    hours of availability, and number of matches they already have.
    :param data_path: path to the directory with the tutor info file
    :param verbose: whether to print out information about minor issues in the input data and how they're handled
    """
    
    tutor_info = pd.read_csv(data_path + 'tutor_info.txt', sep='\t', header=None, index_col=0,
                            names=['id', 'name', 'classes', 'avail_hours', 'n_matches']).sort_index()
    tutor_info.classes = tutor_info.classes.apply(literal_eval)

    n_zero_hours = (tutor_info.avail_hours == 0).sum()
    if n_zero_hours > 0:
        if verbose:
            print("{} tutors had 0 hours available and are thus being dropped.".format(n_zero_hours))
        tutor_info.drop(tutor_info[tutor_info.avail_hours == 0].index, inplace=True)

    max_matches = 3
    n_max_matches = (tutor_info.n_matches >= max_matches).sum()
    if n_max_matches > 0:
        if verbose:
            print("{} tutors had {} matches already and are thus being dropped.".format(n_max_matches, max_matches))
        tutor_info.drop(tutor_info[tutor_info.n_matches == max_matches].index, inplace=True)

    n_no_classes = (tutor_info.classes.apply(len) == 0).sum()
    if n_no_classes > 0:
        if verbose:
            print("{} tutors had an empty class list and are thus being dropped.".format(n_no_classes))
        tutor_info.drop(tutor_info[tutor_info.classes.apply(len) == 0].index, inplace=True)
    return tutor_info


def load_tutee_info(data_path, verbose=False):
    """
    Reads in the tutee data which must be in a file named tutee_info.txt in the directory specified by data_path
    (entered as a command-line argument with default value of the current directory).
    Columns expected in the tutee_info file are tutee id, tutee name, list of classes requested for tutoring and
    number of matches they already have.
    :param data_path: path to the directory with the tutee info file
    :param verbose: whether to print out information about minor issues in the input data and how they're handled
    """
    
    tutee_info = pd.read_csv(data_path + 'tutee_info.txt', sep='\t', header=None, index_col=0,
                            names=['id', 'name', 'classes', 'n_matches']).sort_index()
    tutee_info.classes = tutee_info.classes.apply(literal_eval)

    n_no_classes = (tutee_info.classes.apply(len) == 0).sum()
    if n_no_classes > 0:
        if verbose:
            print("{} tutees had an empty class list and are thus being dropped.".format(n_no_classes))
        tutee_info.drop(tutee_info[tutee_info.classes.apply(len) == 0].index, inplace=True)

    return tutee_info


def get_class_priority_and_mappings(tutor_info, tutee_info):
    """
    Computes the priority assigned to each class as the number of tutees requesting that class divided by the number of tutors available
    for that class (a priority of 0 is always given if a class doesn't have both tutees and tutors)
    Also returns a few useful mappings to/from class names.
    :returns: class_priority: pandas dataframe with columns class_name, priority
              class_to_id: map from class names to ids (as given in the input files)
              class_to_idx: map from class names to indices, which are [0:n_classes] and given in order according to class_priority
              idx_to_class: inverse map of class_to_idx
    """
    
    # extract just a list of names of classes per tutor/tutee; reduce these into one long list; make a Series mapping names to counts
    # then set priority = n_tutees / n_tutors for each class
    tutees_per_class = pd.Series(reduce(lambda x, y: x + y, tutee_info.classes.apply(lambda x: [elem[1] for elem in x]))).value_counts()
    tutors_per_class = pd.Series(reduce(lambda x, y: x + y, tutor_info.classes.apply(lambda x: [elem[1] for elem in x]))).value_counts()
    class_priority = (tutees_per_class / tutors_per_class).fillna(0).reset_index() # NA occurs when a class doesn't have both tutors and tutees
    class_priority.rename(columns={'index': 'class_name', 0: 'priority'}, inplace=True)
    class_priority.sort_values('priority', inplace=True)
    class_priority.priority /= class_priority.priority.sum() # normalize
    
    class_id_name = np.concatenate((tutee_info.classes.map(lambda class_list: [class_elem[:2] for class_elem in class_list]).values,
                                    tutor_info.classes.values))
    class_to_id = {name: idx for (idx, name) in reduce(lambda x, y: x + y, class_id_name)} # id is whatever was in the input file
    class_to_idx = {class_priority.class_name.values[i]: i for i in xrange(len(class_priority))} # idx is [0 : n_classes]
    idx_to_class = {val: key for (key, val) in class_to_idx.items()}
    return class_priority, class_to_id, class_to_idx, idx_to_class


def get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes):
    """
    Computes the index in the 1D array corresponding to (tutor_idx, tutee_idx, class_idx) in the
    imagined 3D tensor
    """
    assert tutor_idx < n_tutors
    assert tutee_idx < n_tutees
    assert class_idx < n_classes
    return tutee_idx + n_tutees * tutor_idx + n_tutees * n_tutors * class_idx


def get_triple_idx(idx, n_tutees, n_tutors):
    """
    Does the inverse of get_idx: returns the (tutor_idx, tutee_idx, class_idx) corresponding to idx
    """
    class_idx = 0
    while idx - (n_tutees * n_tutors) >= 0:
        class_idx += 1
        idx -= (n_tutees * n_tutors)
    
    tutor_idx = 0
    while idx - n_tutees >= 0:
        tutor_idx += 1
        idx -= n_tutees
    tutee_idx = idx
    return tutor_idx, tutee_idx, class_idx


def get_class_list_constraints(tutor_info, tutee_info, n_variables, get_class_idx):
    """
    Computes and returns constraints & bounds that will enforce that no matching occurs between a tutor and tutee unless it
    is in a class which occurs in both of their class lists.
    :returns: class_list_bounds, class_list_constraints (both are numpy arrays that can be passed to scipy.optimize.linprog
    as constraints / bounds, respectively)
    """
    
    n_tutees = len(tutee_info)
    n_tutors = len(tutor_info)
    n_classes = n_variables / (n_tutees * n_tutors)
    
    class_list_bounds = 0
    class_list_constraints = np.ones((1, n_variables))

    # set indices to 0 where the proposed matchings are valid; then any >= 0 value is possible for those matchings
    # the others will be forced to be 0 because we'll constrain their sum to be 0
    for tutor_idx in xrange(n_tutors):
        tutor_class_indices = get_class_idx([elem[1] for elem in tutor_info.classes.iloc[tutor_idx]]) # elem[1] is class name
        for class_idx in tutor_class_indices:
            for tutee_idx in xrange(n_tutees):
                tutee_class_indices = get_class_idx([elem[1] for elem in tutee_info.classes.iloc[tutee_idx]])
                if class_idx in tutee_class_indices:
                    class_list_constraints[0, get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes)] = 0
    
    return class_list_bounds, class_list_constraints


def get_hours_constraints(tutor_info, tutee_info, n_variables, get_class_idx, iterative_matching):
    """
    Computes and returns constraints & bounds that will enforce that no tutor tutors more hours than they have available
    and that no tutee receives more tutoring in a class than they requested.
    :returns: hours_bounds, hours_constraints (both are numpy arrays that can be passed to scipy.optimize.linprog
    as constraints / bounds, respectively)
    """
    n_tutors = len(tutor_info)
    n_tutees = len(tutee_info)
    n_classes = n_variables / (n_tutors * n_tutees)
    
    hours_constraints = []
    hours_bounds = []

    # tutees need one constraint per class (# hours requested is per class)
    for tutor_idx in xrange(n_tutors):
        class_indices = get_class_idx([elem[1] for elem in tutor_info.classes.iloc[tutor_idx]]) # elem[1] is class name
        
        if iterative_matching:
            hours_bounds.append(1)
        else:
            hours_bounds.append(tutor_info.avail_hours.iloc[tutor_idx])
        
        constraint = np.zeros((1, n_variables)) # set indices to 1 where the proposed class is valid for this tutor
        for class_idx in class_indices:
            for tutee_idx in xrange(n_tutees):
                constraint[0, get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes)] = 1
        hours_constraints.append(constraint)

    for tutee_idx in xrange(n_tutees):
        class_indices = get_class_idx([elem[1] for elem in tutee_info.classes.iloc[tutee_idx]]) # elem[1] is class name
        hours_requested = [elem[2] for elem in tutee_info.classes.iloc[tutee_idx]]
        for i in xrange(len(class_indices)):
            class_idx = class_indices[i]
            
            if iterative_matching:
                hours_bounds.append(1)
            else:
                hours_bounds.append(hours_requested[i])
            
            constraint = np.zeros((1, n_variables)) # set indices to 1 where the proposed class is valid for this tutee
            for tutor_idx in xrange(n_tutors):
                constraint[0, get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes)] = 1
            hours_constraints.append(constraint)

    hours_constraints = np.concatenate(hours_constraints, axis=0)
    hours_bounds = np.array(hours_bounds)
    return hours_bounds, hours_constraints


def get_objective(lambda_classes, lambda_students, class_priority, n_tutors, n_tutees, n_classes, get_class_idx, tutee_info):
    """
    Generates an objective function that can be optimized using a linear program.
    What is actually returned is a 1D numpy array whose size is the number of variables.
    Each variable is represented by one index in the array. If we call the array A and the
    variables V, then the function to be optimized is
    sum_i V_i * A_i.
    That is, we maximize the weighted sum of the variables. The variables are implicit as far
    as the optimization is concerned: they are not explicitly encoded; one needs to know what
    each index corresponds to.
    Here, the weights are based on the per-class priorities and whether the students have priority
    for a given class.
    :param lambda_classes: how much weight to put on the per-class priorities. The larger the lambda
                           values are, the more focus is given to high priority classes/students (even
                           at the expense of matching less tutoring hours overall)
    :param lambda_students: how much weight to put on the student priorities for given classes
    :returns: a 1D numpy array where each value is the coefficient for the implicit variable at that
              index
    """
    n_variables = n_tutors * n_tutees * n_classes
    # scale priorities by lambdas
    scaled_class_priorities = lambda_classes * class_priority.priority.values
    objective_function = np.ones(n_variables)

    for class_idx in xrange(n_classes):
        priority = scaled_class_priorities[class_idx]
        for tutor_idx in xrange(n_tutors):
            for tutee_idx in xrange(n_tutees):
                objective_function[get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes)] *= priority

    for tutee_idx in xrange(n_tutees):
        class_indices = get_class_idx([elem[1] for elem in tutee_info.classes.iloc[tutee_idx]]) # elem[1] is class name
        priorities = [elem[3] for elem in tutee_info.classes.iloc[tutee_idx]]
        for i in xrange(len(class_indices)):
            class_idx = class_indices[i]
            priority = .01 + lambda_students * priorities[i] # so priority of 0 -> 1; we don't want to ignore students with no priority
            for tutor_idx in xrange(n_tutors):
                before = objective_function[get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes)]
                objective_function[get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes)] *= priority
                after = objective_function[get_idx(tutor_idx, tutee_idx, class_idx, n_tutees, n_tutors, n_classes)]
    return objective_function


def solve(objective_function, hours_constraints, hours_bounds, class_list_constraints, class_list_bounds, var_bounds, verbose=False):
    """
    Uses a linear program to maximize the given objective function subject to the constraints
    which must be present as global variables: hours_constraints, hours_bounds, class_list_constraint,
    and class_list_bound.
    Attempts first a quick program with fewer bounds. If this fails (in that the solution is outside the desired bounds)
    a slower, completely bounded program is run.
    :param objective_function: a 1D numpy array as specified as the return value of get_objective.
    :returns: A scipy.optimize.OptimizeResult consisting of the following fields:
                x : (numpy ndarray) The independent variable vector which optimizes the linear programming problem.
                slack : (numpy ndarray) The values of the slack variables. Each slack variable corresponds to an inequality
                        constraint. If the slack is zero, then the corresponding constraint is active.
                success : (bool) Returns True if the algorithm succeeded in finding an optimal solution.
                status : (int) An integer representing the exit status of the optimization:
                            0 : Optimization terminated successfully
                            1 : Iteration limit reached
                            2 : Problem appears to be infeasible
                            3 : Problem appears to be unbounded
                nit : (int) The number of iterations performed.
                message : (str) A string descriptor of the exit status of the optimization.
    """
    solution = linprog(-objective_function, options={'disp': verbose},
                       A_ub=hours_constraints, b_ub=hours_bounds,
                       A_eq=class_list_constraints, b_eq=class_list_bounds)
    
    max_hours = var_bounds[1]
    if solution.x.max() > max_hours:
        if verbose:
            print('Quick solution exceeded max_hours ({} hours in a matching; max is {}).'.format(solution.x.max(), max_hours))
            print('Running slower, bounded program.')
        solution = linprog(-objective_function, bounds=var_bounds, options={'disp': verbose},
                           A_ub=hours_constraints, b_ub=hours_bounds,
                           A_eq=class_list_constraints, b_eq=class_list_bounds)
    return solution


def update_info(tutor_info, tutee_info, matching, max_tutees_per_tutor=3):
    """
    Updates tutor_info and tutee_info based on the given matchings:
        Determines the number of hours to assign for each matching
        Removes the appropriate number of hours available from the tutors
        Removes matched classes from the classes list of tutees
        Drops tutors with 0 hours left
        Drops tutees with no classes left
    WARNING: all parameters are modified in place.
    """
    
    for match_idx in xrange(len(matching)):
        tutor_id, tutee_id, class_id = matching.loc[match_idx, ['tutor_id', 'tutee_id', 'class_id']]
        avail_hours = tutor_info.loc[tutor_id].avail_hours
        classes = tutee_info.loc[tutee_id].classes
        request_hours = filter(lambda class_elem: class_elem[0] == class_id, classes)[0][2]

        if request_hours > 4:
            assign_hours = 3
        elif request_hours > 2:
            assign_hours = 2
        else:
            assign_hours = 1

        assign_hours = min(avail_hours, assign_hours)
        matching.loc[match_idx, 'n_hours'] = assign_hours

        hours_left = avail_hours - assign_hours
        n_matches = tutor_info.loc[tutor_id, 'n_matches'] + 1
        if hours_left > 0 and n_matches < max_tutees_per_tutor:
            tutor_info.loc[tutor_id, 'avail_hours'] = hours_left
            tutor_info.loc[tutor_id, 'n_matches'] = n_matches
        else:
            tutor_info.drop(tutor_id, inplace=True)

        classes_left = filter(lambda class_elem: class_elem[0] != class_id, classes)
        if len(classes_left) > 0:
            tutee_info = tutee_info.set_value(tutee_id, 'classes', classes_left) # because value is a list, need this syntax
        else:
            tutee_info.drop(tutee_id, inplace=True)


def get_matching(solution, tutor_info, tutee_info, idx_to_class, class_to_id, lambda_classes, lambda_students,
                 iterative_matching, return_matching=False, verbose=False):
    """
    Converts the solution to the tutor-tutee matching linear program into the desired output file format:
    a tsv with columns ['tutor_id', 'tutor_name', 'tutee_id', 'tutee_name', 'class_id', 'class_name', 'n_hours']
    which specifies all tutor-tutee matchings.
    :param solution: a scipy.optimize.OptimizeResult as returned from scipy.optimize.linprog (e.g. through the solve function)
    :param return_matching: whether to save the matching file to disk (if False) or to return it
    :param verbose: whether to print the name of the matching file.
    """
    
    n_tutees = len(tutee_info)
    n_tutors = len(tutor_info)
    tutor_to_idx = {tutor_info.index.values[i]: i for i in xrange(n_tutors)}
    tutee_to_idx = {tutee_info.index.values[i]: i for i in xrange(n_tutees)}
    idx_to_tutor = {val: key for (key, val) in tutor_to_idx.items()}
    idx_to_tutee = {val: key for (key, val) in tutee_to_idx.items()}
    
    solution.x = solution.x.astype(np.int32)
    
    matched_indices = np.argwhere(solution.x != 0).ravel()
    matches = []
    for matched_idx in matched_indices:
        tutor_idx, tutee_idx, class_idx = get_triple_idx(matched_idx, n_tutees, n_tutors)
        tutor_id = idx_to_tutor[tutor_idx]
        tutor_name = tutor_info.name.loc[tutor_id]
        tutee_id = idx_to_tutee[tutee_idx]
        tutee_name = tutee_info.name.loc[tutee_id]
        class_name = idx_to_class[class_idx]
        class_id = class_to_id[class_name]
        n_hours = solution.x[matched_idx]
        matches.append([tutor_id, tutor_name, tutee_id, tutee_name, class_id, class_name, n_hours])
    matches = pd.DataFrame(matches,
                           columns=['tutor_id', 'tutor_name', 'tutee_id', 'tutee_name', 'class_id', 'class_name', 'n_hours'])
    if return_matching:
        return matches
    else:
        fname = '{}matches_lc_{}_ls_{}_{}.tsv'.format(data_path, lambda_classes, lambda_students,
                                                      'iter' if iterative_matching else 'single')
        matchings.to_csv(fname, sep='\t', index=False)
        print("Saved matching to {}".format(fname))


def main(data_path, max_hours, verbose, use_product, lambda_classes, lambda_students, iterative_matching, return_matching=False):
    """
    :param return_matching: if True, only one matching is computed and then returned (i.e. one can't use multiple lambda values
                            and only one iteration of matching will be done, regardless of the value of iterative_matching)
    """
    
    if use_product:
        lambdas = list(product(lambda_classes, lambda_students))
    else:
        lambdas = zip(lambda_classes, lambda_students)
        
    tutor_info_complete = load_tutor_info(data_path, verbose)
    tutee_info_complete = load_tutee_info(data_path, verbose)
    
    for lambda_idx in xrange(len(lambdas)):
        
        print("\nSolving LP with lambda_classes = {}, lambda_students = {}.".format(*lambdas[lambda_idx]))
    
        tutor_info = tutor_info_complete.copy()
        tutee_info = tutee_info_complete.copy()
        
        matchings = []
        iter_number = 0
        while True:
            iter_number += 1
            if verbose:
                print("\nOn iteration", iter_number)
            
            n_tutors = len(tutor_info)
            n_tutees = len(tutee_info)

            ### class priorities and info
            class_priority, class_to_id, class_to_idx, idx_to_class = get_class_priority_and_mappings(tutor_info, tutee_info)
            n_classes = len(class_priority)
            get_class_idx = np.vectorize(class_to_idx.get)

            ### bounds/constraints on the linear program

            n_variables = n_tutors * n_tutees * n_classes

            var_bounds = (0, max_hours) # same bound for all matchings
            class_list_bounds, class_list_constraints = get_class_list_constraints(tutor_info, tutee_info, n_variables, get_class_idx)
            hours_bounds, hours_constraints = get_hours_constraints(tutor_info, tutee_info, n_variables, get_class_idx, iterative_matching)

            objective_function = get_objective(lambdas[lambda_idx][0], lambdas[lambda_idx][1], class_priority,
                                               n_tutors, n_tutees, n_classes, get_class_idx, tutee_info)
            solution = solve(objective_function, hours_constraints, hours_bounds, class_list_constraints, class_list_bounds, var_bounds,
                            verbose)
            
            if not iterative_matching and not return_matching:
                get_matching(solution, tutor_info, tutee_info, idx_to_class, class_to_id,
                             lambdas[lambda_idx][0], lambdas[lambda_idx][1], iterative_matching, verbose=verbose)
                sys.exit()
            
            if return_matching:
                return get_matching(solution, tutor_info, tutee_info, idx_to_class, class_to_id, lambdas[lambda_idx][0],
                                    lambdas[lambda_idx][1], iterative_matching,verbose=verbose, return_matching=True)
            
            # otherwise: iterative matching and should save final result instead of returning 1 iteration of it
            matching = get_matching(solution, tutor_info, tutee_info, idx_to_class, class_to_id,
                                    lambdas[lambda_idx][0], lambdas[lambda_idx][1], iterative_matching, return_matching=True)
    
            if len(matching) == 0:
                break

            matchings.append(matching)
            update_info(tutor_info, tutee_info, matching)

            if len(tutor_info) == 0 or len(tutee_info) == 0:
                break

        matchings = pd.concat(matchings).reset_index(drop=True)
        
        ### give swap in tutors with 0 matches for those who have multiple
        zero_match_tutors = tutor_info[tutor_info.n_matches == 0]
        
        n_zero_match_tutors = len(zero_match_tutors)
        if verbose:
            if n_zero_match_tutors:
                print("\nFound {} tutors with 0 matches. Attempting to swap them in.".format(n_zero_match_tutors))
            else:
                print("\nNo tutors with 0 matches found.")
        
        for tutor_id in zero_match_tutors.index:
            class_ids = map(lambda class_elem: class_elem[0], zero_match_tutors.loc[tutor_id, 'classes']) # just get the ids
            matching_to_swap = None
            most_matches = 0
            for row in matchings.index:
                if matchings.loc[row, 'class_id'] in class_ids:
                    swapped_tutor_id = matchings.loc[row, 'tutor_id']
                    n_matches = matchings.tutor_id.value_counts().loc[swapped_tutor_id]
                    if n_matches > max(1, most_matches):
                        most_matches = n_matches
                        matching_to_swap = [row, swapped_tutor_id, matchings.loc[row, 'n_hours'], matchings.loc[row, 'tutee_id'],
                                            matchings.loc[row, 'class_id']]

            if matching_to_swap: # not None
                # update the matching to use the zero-match tutor swapped for the old one
                row, swapped_tutor_id, swapped_hours, tutee_id, class_id = matching_to_swap
                hours_requested = filter(lambda class_elem: class_elem[0] == class_id, tutee_info_complete.loc[tutee_id, 'classes'])[0][2]
                hours_matched = min(zero_match_tutors.loc[tutor_id, 'avail_hours'], hours_requested)
                matchings.loc[row, 'tutor_id'] = tutor_id
                matchings.loc[row, 'tutor_name'] = zero_match_tutors.loc[tutor_id, 'name']
                matchings.loc[row, 'n_hours'] = hours_matched

                # also update the tutor info based on the swap; we'll try one more LP, just in case
                tutor_info.loc[tutor_id, 'n_matches'] += 1
                tutor_info.loc[tutor_id, 'avail_hours'] -= hours_matched

                if swapped_tutor_id in tutor_info.index: # edit existing entry
                    tutor_info.loc[swapped_tutor_id, 'n_matches'] -= 1
                    tutor_info.loc[swapped_tutor_id, 'avail_hours'] += swapped_hours
                else: # put an entry back in; make sure to update hours and matchings based on existing matchings
                    tutor_info = tutor_info.append(tutor_info_complete.loc[swapped_tutor_id])
                    tutor_info.loc[swapped_tutor_id, 'n_matches'] = len(matchings.loc[[swapped_tutor_id]])
                    tutor_info.loc[swapped_tutor_id, 'avail_hours'] -= matchings.loc[[swapped_tutor_id], 'n_hours'].sum()

                if tutor_info.loc[tutor_id, 'avail_hours'] == 0:
                    tutor_info.drop(tutor_id, inplace=True)
        
        if n_zero_match_tutors:
            # do one more matching, just in case anybody swapped out could be matched still
            tutor_info.to_csv('tmp/tutor_info.txt', sep='\t', header=None)
            tutee_info.to_csv('tmp/tutee_info.txt', sep='\t', header=None)

            matching = main('tmp/', max_hours, verbose, use_product, [lambdas[lambda_idx][0]], [lambdas[lambda_idx][1]],
                            iterative_matching=True, return_matching=True)
            update_info(tutor_info, tutee_info, matching)
            matchings = pd.concat((matchings, matching))
        
        ### increase the number of hours in matchings where allowable
        hours_remaining = tutor_info_complete.avail_hours - matchings.groupby('tutor_id').n_hours.sum().sort_index()
        hours_remaining = hours_remaining[hours_remaining > 0]
        
        if verbose:
            print("\nExpanding the number of hours for matches where possible.")
        for tutor_id in hours_remaining.index:
            matchings.set_index('tutor_id', inplace=True)
            tutor_info_tmp = tutor_info_complete.loc[[tutor_id]].copy() # selecting with "[]" keeps it a dataframe instead of a series
            tutor_info_tmp.loc[:, 'avail_hours'] = hours_remaining.loc[tutor_id]
            tutee_info_tmp = tutee_info_complete.loc[matchings.loc[[tutor_id]].tutee_id].copy()

            for i in xrange(len(tutee_info_tmp)):
                tutee_id = tutee_info_tmp.index[i]
                matched_class = filter(lambda class_elem: class_elem[0] == matchings.loc[[tutor_id]].iloc[i].class_id,
                                       tutee_info_tmp.iloc[i].classes)
                tutee_info_tmp.set_value(tutee_id, 'classes', matched_class)

            tutor_info_tmp.to_csv('tmp/tutor_info.txt', sep='\t', header=None)
            tutee_info_tmp.to_csv('tmp/tutee_info.txt', sep='\t', header=None)

            matching = main('tmp/', max_hours, verbose, use_product, [lambdas[lambda_idx][0]], [lambdas[lambda_idx][1]],
                            iterative_matching=False, return_matching=True)
            matchings = matchings.reset_index().set_index(['tutor_id', 'tutee_id'])
            
            for row_idx in xrange(len(matching)):
                tutee_id, n_hours = matching.iloc[row_idx].loc[['tutee_id', 'n_hours']]
                matchings.loc[(tutor_id, tutee_id), 'n_hours'] += n_hours
            matchings.reset_index(inplace=True)
        
        assert all(matchings.groupby('class_id').tutee_id.value_counts() == 1)
        assert all(matchings.tutor_id.value_counts() <= 3)
        
        fname = '{}matches_lc_{}_ls_{}_{}.tsv'.format(data_path, lambdas[lambda_idx][0], lambdas[lambda_idx][1],
                                                      'iter' if iterative_matching else 'single')
        matchings.to_csv(fname, sep='\t', index=False)
        print("Saved matching to {}".format(fname))

    if verbose:
        runtime = time.time() - start_time
        print("\nRuntime: {:.0f} seconds".format(runtime))


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-lc', '--lambda_classes', nargs='+', type=float, default=[1],
                        help="The coefficients that determine how much weight is given to prioritizing 'harder'\
                        classes (those with more tutees compared to tutors); must be strictly positive. Default: 1")
    parser.add_argument('-ls', '--lambda_students', nargs='+', type=float, default=[1],
                        help="The coefficients that determine how much weight is given to prioritizing students\
                        in especial need (those marked as priority for a given class); must be strictly positive. Default: 1")
    parser.add_argument('-p', '--data_path', help="Path to the input files (tutee_info.txt and tutor_info.txt). Default: ./",
                        default='./')
    parser.add_argument('-m', '--max_hours', help="Maximum number of hours allowable in one match (if doing iterative matching, this\
                        will always be set to 1). Default: 3", type=int, default=3)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Whether to print addition information while running. Default: False", default=False)
    parser.add_argument('-sm', '--single_matching', action='store_true', help="Whether to use iterative matching (see notebook for\
                        more on this approach) or single matching; Default: False (i.e. use iterative matching).", default=False)
    parser.add_argument('-prod', '--cartesian_product', action='store_true', default=False,
                       help="If this flag is given, one matching is computed for each combination of\
                        lambda_classes and lambda_students. Otherwise, the two are zipped. Example: if\
                        lambda_students = [2, 5] and lambda_classes = [2, 3] then without this flag, 2 matchings\
                        will be computed with lambdas: (2, 2) and (5, 3). With this flag set, 4 matchings will\
                        be computed: (2, 2), (2, 3), (5, 2), (5, 3). Default: False.")
    
    args = parser.parse_args()
    data_path = args.data_path
    verbose = args.verbose
    use_product = args.cartesian_product
    lambda_classes = args.lambda_classes
    lambda_students = args.lambda_students
    iterative_matching = not args.single_matching
    
    if iterative_matching:
        max_hours = 1
        
    if verbose:
        print("Data path:", data_path)
        print("lambda_students:", lambda_students)
        print("lambda_classes:", lambda_classes)
        print("Use Cartesian product of lambdas?", use_product)
        print("max_hours:", max_hours, end='\n\n')
        print("iterative_matching?", iterative_matching)

    main(data_path, max_hours, verbose, use_product, lambda_classes, lambda_students, iterative_matching)