# -*- encoding: utf-8 -*-
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""

def find_optimal_n_components(var_ratio, goal_var):
    """
    Find optimal n_components for truncated svd given a variance threshold

    :param var_ratio list:
        List of explained variance per n_component

    :param goal_var float:
        Desired variance threshold

    :return: Optimal n_components for a given variance threshold

    """
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components
