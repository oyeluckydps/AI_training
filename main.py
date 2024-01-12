""""
This is a Python script for experimenting with local search algorithms.
The challenge is to place n points on a d dimensional hypercube such that:
    1.) The manhattan distance between any two pair is maximized.
    2.) The sum of manhattan distances among all possible pairs is maximized.
"""
import copy
import random
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

random_seed = 10
random.seed(random_seed)
print("Initiated with random seed = ", random_seed)


def min_distance(mat):
    """
    :param mat: A boolean matrix for which each row corresponds to a d-dimension bool entry for a point.
    :return: Minimum MH distance between any two points.
    """
    num_rows = np.size(mat, 0)
    MH_distance = [np.sum(np.bitwise_xor(mat[i], mat[j])) for i in range(num_rows) for j in range(i+1, num_rows)]
    return -min(MH_distance)

def sum_distance(mat):
    """
    :param mat: A boolean matrix for which each row corresponds to a d-dimension bool entry for a point.
    :return: Sum of MH distance among all pairs of two points.
    """
    num_rows = np.size(mat, 0)
    MH_distance = [np.sum(np.bitwise_xor(mat[i], mat[j])) for i in range(num_rows) for j in range(i+1, num_rows)]
    return -sum(MH_distance)/num_rows


def combined_distance(mat):
    """
    The function comnines the MH distance and min distance cost function by providing large weightage to min distance.
    :param mat: A boolean matrix for which each row corresponds to a d-dimension bool entry for a point.
    :return: Sum of MH distance + 100 * min_distance
    """
    FACTOR = 500
    return min_distance(mat)*FACTOR + sum_distance(mat)

def find_nearby_points(mat, cost_fun, T):
    """
    Find all the points (rows) that are near to other points and are problematic.
    :param mat: A boolean matrix for which each row corresponds to a d-dimension bool entry for a point.
    :param cost_fun: Cost function.
    :param T: Determines how many of the unique maximum cost to be replaced.
    :return: A list of row index corresponding to problematic points.
    """
    num_rows = np.size(mat, 0)
    MH_distance= {}
    max_dis = float('-inf')
    for i in range(num_rows):
        for j in range(num_rows):
            if i==j:
                continue
            temp_mat = np.vstack([mat[i], mat[j]])          # Make a temporary matrix with only two points and pass it to cost function.
            dis = cost_fun(temp_mat)
            if dis > max_dis:                               # If the cost turns out to be more than maximum cost then update the max_dis
                max_dis = dis
            MH_distance[(i, j)] = dis
    problematic_points = [i for k, v in MH_distance.items() for i in k if v == max_dis]         # Find the points that occur in maximum cost function pairs.
    # unique_points = list(set(problematic_points))
    # frequency_of_problematic_points = [problematic_points.count(i)/2 for i in unique_points]    # Find the frequency of each of those problematic points.
    # frequency_of_problematic_points, unique_points = zip(*sorted(zip(frequency_of_problematic_points, unique_points), reverse=True))    # Sort the problematic points in order of their occurence.
    # number_of_most_problematic_points = frequency_of_problematic_points.count(frequency_of_problematic_points[0])                       # Count number of problematic points that occur most number of times.
    # most_problematic_points = unique_points[:number_of_most_problematic_points]                 # Return the points creating problem most number of times.
    return set(problematic_points)


def replace_points(mat, points):
    """
    The function replaces all the problematic points with another random point.
    :param mat: A boolean matrix for which each row corresponds to a d-dimension bool entry for a point.
    :param points: LIst of points/rows to be replaced
    :return: New matrix after the replacement of the rows.
    """
    new_mat = copy.deepcopy(mat)
    for row in points:
        new_mat[row] = np.random.randint(2, size=(1, 7))
    return new_mat


def descent_points(mat, points, cost_fun):
    """
    The function replaces one of the problematic point with the point's neighbour such that the cost function is minimized.
    :param mat: A boolean matrix for which each row corresponds to a d-dimension bool entry for a point.
    :param points: List of points/rows to be replaced
    :param cost_fun: Cost function.
    :return: New matrix after the replacement of the rows.
    """
    new_mat = copy.deepcopy(mat)
    base_cost = cost_fun(mat)
    all_cost = []
    all_combos = []
    for row in points:
        for idx, val in enumerate(mat[row]):
            new_mat[row][idx] = (new_mat[row][idx] + 1)%2
            current_cost = cost_fun(new_mat)
            all_combos.append((row, idx))
            all_cost.append(current_cost)
            new_mat[row][idx] = (new_mat[row][idx] + 1) % 2
    all_cost_sorted, all_combos_sorted = zip(*sorted(zip(all_cost, all_combos)))
    for combo in all_combos_sorted:
        row = combo[0]
        idx = combo[1]
        new_mat[row][idx] = (new_mat[row][idx] + 1) % 2
        current_cost = cost_fun(new_mat)
        if current_cost>base_cost:
            return (new_mat, current_cost)
    return (mat, base_cost)


def gradient_descent(mat, cost_fun):
    """
    A function that tries to optimize the mat pertaining to the cost function.
    :param mat: A boolean matrix for which each row corresponds to a d-dimension bool entry for a point.
    :param cost_fun: The cost function that has to be minimized.
    :return: The optimized mat and optimized cost function.
    """
    ITERATIONS = 100
    for i in range(ITERATIONS):
        logging.debug(i)
        base_cost = cost_fun(mat)
        logging.debug(base_cost)
        problem_points = find_nearby_points(mat, cost_fun, 1)
        # T = 1
        # while True:
        #     new_problem_points = find_nearby_points(mat, cost_fun, T)        # Find all the nearby points (one or more) that might need to be replaced.
        #     if len(new_problem_points.difference(problem_points)) == 0:
        #         T = T+1
        #     else:
        #         problem_points = new_problem_points
        #         break
        logging.debug(problem_points)

        # new_mat = replace_points(mat, problem_points)       # An algorithm to replace all the points that are creating problem as they are nearby.
        new_mat, new_cost = descent_points(mat, problem_points, cost_fun)
        if new_cost <= base_cost:
            mat = new_mat
        logging.debug(new_cost)
    return mat, cost_fun(mat)


def main():
    """
    All the universal variables, calls to other relevant functions and print of the output goes in here.
    Variable Description:
        n   --> Number of points.
        d   --> Dimensions of the hypercube.
        cost_fun    --> Reference to the cost function pertaining to 1st or 2nd challenge.
    """
    n = 16
    d = 7
    cost_fun = combined_distance

    init_mat = np.random.randint(2, size=(n, d))

    final_matrix, least_cost = gradient_descent(init_mat, cost_fun)
    print(" Final matrix is")
    print(final_matrix)
    print("The minimized cost function is = ", least_cost)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
