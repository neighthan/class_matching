#!/usr/bin/python

from __future__ import division, print_function
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from ast import literal_eval
from collections import Counter
import argparse
from itertools import product
import time


def load_tutor_info(verbose=True):
    tutor_info = pd.read_csv(data_path + 'tutor_info.txt', sep='\t', index_col=0).sort_index()
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


def load_tutee_info(verbose=True):
    tutee_info = pd.read_csv(data_path + 'tutee_info.txt', sep='\t', index_col=0).sort_index()
    tutee_info.classes = tutee_info.classes.apply(literal_eval)

    n_no_classes = (tutee_info.classes.apply(len) == 0).sum()
    if n_no_classes > 0:
        if verbose:
            print("{} tutees had an empty class list and are thus being dropped.".format(n_no_classes))
        tutee_info.drop(tutee_info[tutee_info.classes.apply(len) == 0].index, inplace=True)

    return tutee_info


def get_idx(tutor_idx, tutee_idx, class_idx):
    """
    Computes the index in the 1D array corresponding to (tutor_idx, tutee_idx, class_idx) in the
    imagined 3D tensor
    """
    assert tutor_idx < n_tutors
    assert tutee_idx < n_tutees
    assert class_idx < n_classes
    return tutee_idx + n_tutees * tutor_idx + n_tutees * n_tutors * class_idx


def get_triple_idx(idx):
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


def get_objective(lambda_classes, lambda_students):
    # scale priorities by lambdas
    scaled_class_priorities = lambda_classes * class_priority.priority.values
    objective_function = np.ones(n_variables)

    for class_idx in xrange(n_classes):
        priority = scaled_class_priorities[class_idx]
        if priority > 1:
            print(priority)
        for tutor_idx in xrange(n_tutors):
            for tutee_idx in xrange(n_tutees):
                objective_function[get_idx(tutor_idx, tutee_idx, class_idx)] *= priority

    for tutee_idx in xrange(n_tutees):
        class_indices = get_class_idx([elem[1] for elem in tutee_info.classes.iloc[tutee_idx]]) # elem[1] is class name
        priorities = [elem[3] for elem in tutee_info.classes.iloc[tutee_idx]]
        for i in xrange(len(class_indices)):
            class_idx = class_indices[i]
            priority = lambda_students * (1 + priorities[i]) # so priority of 0 -> 1
            for tutor_idx in xrange(n_tutors):
                before = objective_function[get_idx(tutor_idx, tutee_idx, class_idx)]
                objective_function[get_idx(tutor_idx, tutee_idx, class_idx)] *= priority
                after = objective_function[get_idx(tutor_idx, tutee_idx, class_idx)]
    return objective_function


def solve(objective_function):
    solution = linprog(-objective_function, options={'disp': True},
                       A_ub=hours_constraints, b_ub=hours_bounds,
                       A_eq=class_list_constraint, b_eq=class_list_bound)
    if solution.x.max() > max_hours:
        print('Quick solution exceeded max_hours ({} hours in a matching; max is {}).'.format(solution.x.max(), max_hours))
        print('Running slower, bounded program.')
        solution = linprog(-objective_function, bounds=(0, max_hours), options={'disp': True},
                           A_ub=hours_constraints, b_ub=hours_bounds,
                           A_eq=class_list_constraint, b_eq=class_list_bound)
    return solution


def save_matching(solution, i, verbose=False):
    """
    :param solution: returned from scipy.optimize.linprog
    :param i: number of this matching; used to name the saved file
    """
    solution.x = solution.x.astype(np.int32)
    
    matched_indices = np.argwhere(solution.x != 0).ravel()
    matches = []
    for matched_idx in matched_indices:
        tutor_idx, tutee_idx, class_idx = get_triple_idx(matched_idx)
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
    matches.to_csv(data_path + 'matches_{}.tsv'.format(i), sep='\t', index=False)
    if verbose:
        print("Saved matching to", data_path + 'matches_{}.tsv'.format(i))


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-lc', '--lambda_classes', nargs='+', type=float, required=True,
                        help="The coefficients that determine how much weight is given to prioritizing 'harder'\
                        classes (those with more hours of tutoring needed).")
    parser.add_argument('-ls', '--lambda_students', nargs='+', type=float, required=True,
                        help="The coefficients that determine how much weight is given to prioritizing students\
                        in need (those marked as higher priority).")
    parser.add_argument('-p', '--data_path', help="Path to the input files (tutee_info.txt and tutor_info.txt).")
    parser.add_argument('-m', '--max_hours', help="Maximum number of hours allowable in one match.", type=int)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Whether to print addition information while running.")
    parser.add_argument('-prod', '--cartesian_product', action='store_true',
                       help="If this flag is given, one matching is computed for each combination of\
                        lambda_classes and lambda_students. Otherwise, the two are zipped. Example: if\
                        lambda_students = [2, 5] and lambda_classes=[2, 3] then without this flag, 2 matchings\
                        will be computed with lambdas: (2, 2) and (5, 3). With this flag set, 4 matchings will\
                        be computed: (2, 2), (2, 3), (5, 2), (5, 3).")
    
    args = parser.parse_args()
    data_path = args.data_path if args.data_path is not None else './'
    max_hours = args.max_hours if args.max_hours is not None else 3
    verbose = args.verbose if args.verbose is not None else False
    use_product = args.cartesian_product if args.cartesian_product is not None else False
    lambda_classes = args.lambda_classes
    lambda_students = args.lambda_students
    
    if verbose:
        print("Data path:", data_path)
        print("lambda_students:", lambda_students)
        print("lambda_classes:", lambda_classes)
        print("Use Cartesian product of lambdas?", use_product)
        print("max_hours:", max_hours, end='\n\n')

    tutor_info = load_tutor_info()
    n_tutors = len(tutor_info)
    tutor_to_idx = {tutor_info.index.values[i]: i for i in xrange(n_tutors)}
    idx_to_tutor = {val: key for (key, val) in tutor_to_idx.items()}

    tutee_info = load_tutee_info()
    n_tutees = len(tutee_info)
    tutee_to_idx = {tutee_info.index.values[i]: i for i in xrange(n_tutees)}
    idx_to_tutee = {val: key for (key, val) in tutee_to_idx.items()}
    
    class_id_name = np.concatenate((tutee_info.classes.map(lambda class_list: [class_elem[:2] for class_elem in class_list]).values, tutor_info.classes.values))
    class_to_id = {name: idx for (idx, name) in reduce(lambda x, y: x + y, class_id_name)}

    # the lists in classes have elements (class_id, class_name, hours_requested, priority)
    # find the total number of hours requested per class by making a dict: {name: hours} for each row, then convert
    # to a Counter so that you can sum them all together into one

    hours_per_class_list = tutee_info.classes.apply(lambda x: {elem[1]: elem[2] for elem in x}).values
    class_priority = sum((Counter(d) for d in hours_per_class_list), Counter())

    for class_ in class_to_id:
        if class_ not in class_priority:
            class_priority[class_] = 0 # no tutees requested this class, though there are tutors available

    class_priority = pd.DataFrame(class_priority.items(), columns=['class_name', 'priority']).sort_values('class_name')
    class_priority.priority /= class_priority.priority.sum()

    n_classes = len(class_priority)
    class_to_idx = {class_priority.class_name.values[i]: i for i in xrange(n_classes)}
    idx_to_class = {val: key for (key, val) in class_to_idx.items()}
    get_class_idx = np.vectorize(class_to_idx.get)
    
    n_variables = n_tutors * n_tutees * n_classes
    var_bounds = (0, max_hours) # same bounds for all variables
    
    class_list_bound = 0
    class_list_constraint = np.ones((1, n_variables)) # set indices to 0 where the proposed matchings are valid

    for tutor_idx in xrange(n_tutors):
        tutor_class_indices = get_class_idx([elem[1] for elem in tutor_info.classes.iloc[tutor_idx]]) # elem[1] is class name
        for class_idx in tutor_class_indices:
            for tutee_idx in xrange(n_tutees):
                tutee_class_indices = get_class_idx([elem[1] for elem in tutee_info.classes.iloc[tutee_idx]])
                if class_idx in tutee_class_indices:
                    class_list_constraint[0, get_idx(tutor_idx, tutee_idx, class_idx)] = 0

    # hours requested/available bounds
    # similar to above but tutees need one constraint per class (# hours requested is per class)

    hours_constraints = []
    hours_bounds = []

    for tutor_idx in xrange(n_tutors):
        class_indices = get_class_idx([elem[1] for elem in tutor_info.classes.iloc[tutor_idx]]) # elem[1] is class name
        hours_bounds.append(tutor_info.avail_hours.iloc[tutor_idx])
        constraint = np.zeros((1, n_variables)) # set indices to 1 where the proposed class is valid for this tutor
        for class_idx in class_indices:
            for tutee_idx in xrange(n_tutees):
                constraint[0, get_idx(tutor_idx, tutee_idx, class_idx)] = 1
        hours_constraints.append(constraint)

    for tutee_idx in xrange(n_tutees):
        class_indices = get_class_idx([elem[1] for elem in tutee_info.classes.iloc[tutee_idx]]) # elem[1] is class name
        hours_requested = [elem[2] for elem in tutee_info.classes.iloc[tutee_idx]]
        for i in xrange(len(class_indices)):
            class_idx = class_indices[i]
            hours_bounds.append(hours_requested[i])
            constraint = np.zeros((1, n_variables)) # set indices to 1 where the proposed class is valid for this tutee
            for tutor_idx in xrange(n_tutors):
                constraint[0, get_idx(tutor_idx, tutee_idx, class_idx)] = 1
            hours_constraints.append(constraint)

    hours_constraints = np.concatenate(hours_constraints, axis=0)
    hours_bounds = np.array(hours_bounds)
    
    if use_product:
        lambdas = list(product(lambda_classes, lambda_students))
    else:
        lambdas = zip(lambda_classes, lambda_students)
    
    for i in xrange(len(lambdas)):
        if verbose:
            print("\nSolving LP with lambda_classes = {}, lambda_students = {}.".format(*lambdas[i]))
        objective_function = get_objective(*lambdas[i])
        solution = solve(objective_function)
        save_matching(solution, i, verbose)
    
    if verbose:
        runtime = time.time() - start_time
        print("\nRuntime: {:.0f} seconds".format(runtime))