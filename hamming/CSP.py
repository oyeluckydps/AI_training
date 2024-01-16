"""
This is a AI program to develop a Constraint Satisfaction Problem.
The challenge is to place n points on a d-dimension hypercube such that distance between any two points is not less
than t.
We would try to add multiple algorithms related to CSP.

5.2) Constraint Consistency: We will apply an algorithm similar to AC-3 to (but for n-ary constraint) ensure
        consistency among all constraints.

5.3) Variable assignment: Assign variable in form of a search tree to find the point distribution.
    5.3.1) Apply MRV (Minimum Remaining Values) choosing the variable with lowest number of elements in domain
                to get early failure.
    and  ) Choose the value for a variable that constrains other variables least.
    5.3.2) Check for consistency after variable assignment.
    5.3.3) We will skip intelligent backtracking for this case as the expected gain shall already be covered by
                consistency check.
    5.3.4) Subtree removal might not be very relevant here but we may divide the constraints into two sets. One that
            is easy to check which will act as subgraph removal.

5.4) Constraint weighting using local search is already partially tried in the local_search.py. However, plateau search
may be added to the code for enhancement.

5.5) Decomposition into tree is also next to impossible for this problem. However, Value Symmetry can be exploited by
setting the initial two points. It can be further exploited by adding to the code that the order of rows or columns
is not important. Furthermore, all rows may be XORed with a boolean string of d-dimension.
"""

import copy
import random
import numpy as np
import logging
from itertools import product

logging.basicConfig(level=logging.DEBUG)

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
print("Initiated with random seed = ", random_seed)


class Variable:
    """
    This class handles all the function sand state of a single variable.
    """

    def __init__(self, d, domain=None):
        """
        :param d: Dimensions of the Variable
        :param domain: Initial domain. If not provided, use range(2**d) as the domain.
        """
        self.d = d
        self.__domain = copy.deepcopy(domain) if domain is not None else list(range(2**d))
        self._enum_domain = []
        self.reset()

    def tup_to_dec(self, bool_list):
        """
        Convert a list of boolean into its equivalent decimal integer.
        """
        powered = [elem*2**(self.d-idx-1) for idx, elem in enumerate(bool_list)]
        return sum(powered)

    def dec_to_tup(self, val):
        """
        Convert a decimal number into its equivalent list of booleans.
        """
        return tuple([(val >> (i-1)) % 2 for i in range(self.d, 0, -1)])

    def reset(self, expand_domain=False):
        """
        Reset the object and set the domain to initial domain.
        """
        self._enum_domain = []
        if expand_domain:
            self.__domain = list(range(2**self.d))
        for val in self.__domain:
            self.add_to_domain(val)

    def add_to_domain(self, val):
        """
        Add a value to domain.
        """
        if isinstance(val, tuple):
            val = self.tup_to_dec(val)
        if val not in self._enum_domain:
            self._enum_domain.append(val)

    def remove_from_domain(self, val):
        """
        Remove the provided value from domain enum.
        :return: Raise ValueType Error if the val is not present in the domain.
        Returns False if the domain enum becomes empty post the removal of the val.
        Returns True otherwise.
        """
        if isinstance(val, tuple):
            val = self.tup_to_dec(val)
        try:
            self._enum_domain.remove(val)
        except ValueError:
            raise
        if len(self._enum_domain) == 0:
            return False
        return True

    @property
    def domain(self):
        """
        :return: A list of all values in the domain.
        """
        return [self.dec_to_tup(val) for val in self._enum_domain]

    @property
    def inconsistent(self):
        """
        :return: True if the domain has been reduced to [] and variable is inconsistent with others.
        """
        return len(self._enum_domain) == 0

    @property
    def assigned(self):
        """
        :return: True if there is a single value in the domain of the variable ensuring that the value has been
        assigned.
        """
        return len(self._enum_domain) == 1

    def next_val(self):
        """
        An iteration over the values in domain.
        """
        for val in self._enum_domain:
            yield self.dec_to_tup(val)

    def assign_domain(self, domain):
        """
        Assign the argument as the domain of Variable.
        """
        self._enum_domain = []
        for d in domain:
            self.add_to_domain(d)

    def assign_value(self, val):
        """
        Assign a single value ot the variable after it has collapsed to single solution.
        """
        if isinstance(val, tuple):
            val = self.tup_to_dec(val)
        self._enum_domain = [val]

    def __len__(self):
        """
        :return: THe size of domain of variable
        """
        return len(self._enum_domain)


class Constraint:
    """
    The Constraint class is helpful in maintaining all the constraint equations (saved in self.constraint_function,
    the required variables (saved in self.vars), other arguments required by the constraint function (saved in self.args
    and self.kwargs).
    Then the class supports multiple function for domain reduction, validity checking etc.
    """
    t = None

    def __init__(self, constraint_function, var_count, *args, **kwargs):
        """
        The input params are defined above.
        """
        self.constraint_function = constraint_function
        self.var_count = var_count
        self.vars = args[0:var_count]
        self.args = args[var_count:]
        self.kwargs = kwargs

    def satisfied_through(self):
        """
        :return: An iter over tuple of values from relevant variables (generated through cartesian product)
        that satisfy the constraint.
        """
        domains = [var.domain for var in self.vars]
        value_iter = product(*domains)
        for values in value_iter:
            if self.constraint_function(*values, *self.args, **self.kwargs):
                # yield values if len(values)>1 else yield values[0]
                yield values

    def reduce_domain(self):
        """
        This function reduces the domain of all the variables that it is involved with. Any value in the domain of
        variables that doesn't satisfy the constraint for all possible combinations of the values from other variables
        is removed.
        :return: None if the constrain cannot be satisfied.
        List of list. Corresponding to each variable the list contains all the values from the domain that
        have been removed.
        """
        all_possible_vals = list(self.satisfied_through())      # List of all entries (in form of list) from var1, var2,
                                                                # var3, ... that satisfies the constraint.
        reduced_dom_of_vars = []
        if len(all_possible_vals) == 0:
            for var in self.vars:
                reduced_dom_of_vars.append(var.domain)
                var.assign_domain([])
            return reduced_dom_of_vars

        list_of_possible_vals = list(map(list, zip(*all_possible_vals)))
        # [List of permissible values in var1, List of permissible values in var2, ...]

        for vals, var in zip(list_of_possible_vals, self.vars):     # For each var, find the set of unique vals in its
            # domain subject to the constraints and change the domain to the same.
            reduced_dom_of_vars.append(list(set(var.domain).difference(set(vals))))
            var.assign_domain(list(vals))       # The domain handling in variable takes care than the same element
            # is not repeated in the domain.
        return reduced_dom_of_vars

    @classmethod
    def more_than_t_away(cls, val1, val2):
        """
        The function checks if the values for point have more than or equal to t values in their
        list different, else it is a conflict.
        """
        conflict = [x ^ y for (x, y) in zip(val1, val2)]
        return sum(conflict) >= cls.t

    @staticmethod
    def equal(val1, coordinates):
        """
        This function puts the constraint that the point1 must have the provided coordinates, else it returns 0.
        """
        conflict = [x ^ y for (x, y) in zip(val1, coordinates)]
        return not any(conflict)

    @classmethod
    def less_than_t_conflicts(cls, var1, var2=None):
        if var2 is None:
            # Variable1 is a matrix realization and we have to check for consistency among all rows.
            for idx1, row1 in enumerate(var1):
                for row2 in var1[idx1+1:]:
                    if not cls.more_than_t_away(row1, row2):
                        return False
            return True

        if isinstance(var2, int):
            # Variable1 is a matrix realization and we have to check consistency for the var2'th row value assignment
            # only.
            for idx, row in enumerate(var1):
                if idx == var2:
                    continue
                if not cls.more_than_t_away(var1[var2], row):
                    return False
            return True

        # Assume var1 and var2 to be two values assigned to points and check the conflict
        return cls.more_than_t_away(var1, var2)


class CSP:
    """
    A class to maintain the Problem and all associated algorithms.
    """
    def __init__(self, n, d, t):
        self.n = n
        self.d = d
        self.t = t
        domain = list(range(2**d))                                      # Domain for each variable
        self.all_vars = [Variable(d, domain) for _ in range(n)]         # List of all Variables by initializing.
        self.all_vars[0].assign_value(0)
        self.all_vars[1].assign_value(70)
        self.all_vars[2].assign_value(39)
        self.all_vars[3].assign_value(97)
        self.all_vars[4].assign_value(21)
        self.ASSIGNED           # Set to true if the CSP has been resolved. It may be consistent or inconsistent.
        self.CONSISTENT         # Set to true if it is consistent, False if not, and None if not assigned.
        self.all_constraints = dict()                       # Dict to contain all the constraint equations.
        self.constraint_eqn_for_var = [[] for _ in range(len(self.all_vars))]
        # Put the keys of constraint equations dict that are relevant for the variable on the same index.
        self.construct_constraints()

    @property
    def ASSIGNED(self):
        if any([len(var) == 0 for var in self.all_vars]) or all([len(var) == 1 for var in self.all_vars]):
            return True
        else:
            return False

    @property
    def CONSISTENT(self):
        if not self.ASSIGNED:
            return None
        if any([len(var) == 0 for var in self.all_vars]):
            return False
        else:
            return True

    def print_status(self):
        print("Printing the status of the CSP")
        inconsistencies = [var.inconsistent for var in self.all_vars]
        if self.CONSISTENT is False:
            print("The CSP leads to INCONSISTENCY!")
            print(inconsistencies)
            return
        if self.CONSISTENT is True:
            print("The CSP is CONSISTENT!")
        for var in self.all_vars:
            print("The variable " + str(var), end="")
            if var.assigned:
                print("is assigned to single value: " + str(var.domain[0]))
            else:
                print("has " + str(len(var)) + " values that are:" + str(var.domain))
        print()
        print("There are total of " + str(len(list(self.all_constraints.keys()))) + " consistency equations.")

    def construct_constraints(self):
        """
        Put all the constraint equations to the all_constraints dict over here.
        :return:
        """
        for idx1, var1 in enumerate(self.all_vars):
            for idx2, var2 in enumerate(self.all_vars):
                if idx2 > idx1:
                    self.all_constraints[(idx1, idx2)] = Constraint(Constraint.more_than_t_away, 2, var1, var2)
                    if (idx1, idx2) not in self.constraint_eqn_for_var[idx1]:
                        self.constraint_eqn_for_var[idx1].append((idx1, idx2))
                    if (idx1, idx2) not in self.constraint_eqn_for_var[idx2]:
                        self.constraint_eqn_for_var[idx2].append((idx1, idx2))

    def ac_measure(self):
        """
        A function to measure how well has Arc Consistency algorithm worked.
        :return: -(Sm of number of elements remaining in domain of variables).
        """
        return sum([len(var) for var in self.all_vars])

    def MH_distance(self, fixed_points, val):
        """
        :param fixed_points: A list of points from which the distance of val is to be found.
        :param val: Another point
        :return: Sum of Manhattan Distance of val point from all the fixed poitns.
        """
        MH_distance = [sum([i^j for i, j in zip(row1, row2)]) \
                           for idx1, row1 in enumerate(fixed_points) \
                           for idx2, row2 in enumerate(fixed_points[idx1+1:])]
        return sum(MH_distance)

    def arc_consistency(self):
        """
        A function to check the consistency among all the values in domain of all variables to ensure they are
        consistent i.e. not removable.
        :return: True if arc consistency is maintained. False if domain of a variable is reduced to [].
        """
        if self.ASSIGNED is True:
            return None
        ITERATIONS = 0
        FACTOR = 0.8        # The decay factor for the priorities in each loop.
        PRIORITY_VAL = 1    # The added value for the priorities in each loop.
        all_constraints = list(self.all_constraints.keys())                # A list of all constraint to be checked.
        constraints_priority = [PRIORITY_VAL*FACTOR]*(len(all_constraints))
        # Provide higher priority to equations that are expected to reduce the domain for sure.
        while any(constraints_priority):
            ITERATIONS += 1
            # logging.debug(ITERATIONS)
            # logging.debug(len([priority > 0 for priority in constraints_priority]))
            # logging.debug([len(domain) for domain in self.all_vars])
            current_constraint_idx = constraints_priority.index(max(constraints_priority))
            current_constraint_key = all_constraints[current_constraint_idx]
            constraint = self.all_constraints[current_constraint_key]
            reduction = constraint.reduce_domain()      # Reduce a constraint.
            constraints_priority[current_constraint_idx] = 0
            constraints_priority = [priority*FACTOR for priority in constraints_priority]
            # logging.debug(current_constraint_key)
            for var, reduced_vals in zip(constraint.vars, reduction):
                if len(reduced_vals) > 0:               # If any of the variables involved in this constraint is reduced
                    var_idx = self.all_vars.index(var)
                    for related_constraint in self.constraint_eqn_for_var[var_idx]:     # Added PRIORITY_VAL to the
                        # Priority List for the constraint equation having an affected variable.
                        constraint_idx = all_constraints.index(related_constraint)
                        constraints_priority[constraint_idx] += PRIORITY_VAL
                    pass
            pass
            constraints_priority[current_constraint_idx] = 0    # Current constraint is already checked.
            if self.ASSIGNED is True:
                return None
        return self.ac_measure()

    def find_best_variable(self, variable=None):
        """
        Find the best variable over which to iterate.
        :param variable: a list of variables from which the best variable has to be found.
        :return:
        """
        if self.ASSIGNED is True:
            return None
        variables_to_check = []
        if variable is not None:
            for var in variable:
                if not var.assigned:
                    variables_to_check.append(var)
        else:
            for var in self.all_vars:
                if not var.assigned:
                    variables_to_check.append(var)
        if len(variables_to_check) < 1:
            return None
        elements_in_domain = [len(var.domain) for var in variables_to_check]
        min_elements_idx = elements_in_domain.index(min(elements_in_domain))
        return variables_to_check[min_elements_idx]

    def find_best_value(self, variable=None):
        """
        Find the best value from the domain of a domain if domain is provided. If variable is not provided
        then search for the best variable first.
        Currently supporting the best value finding based on the arc_consistency result and random value assignment.
        :param variable:
        :return:
        """
        IMPLEMENTATION = 'RANDOM'        # ARC_CONSISTENCY, RANDOM, or HEURISTIC
        if variable is None:
            variable = self.find_best_variable()
            if variable is None:
                return None, None, None

        if IMPLEMENTATION == 'ARC_CONSISTENCY':
            best_measure_so_far = float('-inf')
            best_step_so_far = None
            best_val_so_far = None
            for val in variable.next_val():
                copy_for_ac = self.variable_assignment(variable, val)
                ac_measure = copy_for_ac.arc_consistency()
                if ac_measure > best_measure_so_far:
                    best_measure_so_far = ac_measure
                    best_step_so_far = copy_for_ac
                    best_val_so_far = val
            return variable, best_val_so_far, best_step_so_far

        if IMPLEMENTATION == 'HEURISTIC':
            fixed_points = [var.domain[0] for var in self.all_vars if var.assigned]
            best_measure_so_far = float('-inf')
            best_step_so_far = None
            best_val_so_far = None
            for val in variable.next_val():
                copy_for_ac = self.variable_assignment(variable, val)
                ac_measure = copy_for_ac.MH_distance(fixed_points, val)
                if ac_measure > best_measure_so_far:
                    best_measure_so_far = ac_measure
                    best_step_so_far = copy_for_ac
                    best_val_so_far = val
            return variable, best_val_so_far, best_step_so_far

        if IMPLEMENTATION == 'RANDOM':
            random_value = random.choice(variable.domain)
            copy_post_assignment = self.variable_assignment(variable, random_value)
            return variable, random_value, copy_post_assignment

    def variable_assignment(self, variable, value):
        """
        This function assigns the provided value to the variable if the value is in domain. It doesn't return the self
        but a copy to which the assignment has taken place.
        :rtype: (Variable, tuple, CSP)
        :return: None
        """
        variable_idx = self.all_vars.index(variable)
        assert value in variable.domain
        copy_for_assignment = copy.deepcopy(self)
        copy_for_assignment.all_vars[variable_idx].assign_value(value)
        # If the value and variable are known then assign and return.
        return copy_for_assignment


    def search(self, tree_so_far = None):
        """
        A search algorithm on the CSP problem posed.
        :return: self with values assigned and CONSISTENT flag set to True if the problem is solved,
        else with CONSISTENT flag set to False.
        """
        self.arc_consistency()      # Check the arc consistency
        if self.ASSIGNED:           # If the variables are assigned, be it consistent or inconsistent, then return.
            return
        best_var = self.find_best_variable()        # If values are not assigned then it is always possible to find a
                                                    # best variable for this level.
        ITERATION = 0
        while len(best_var) > 0:
            ITERATION += 1
            _, best_val, resulting_CSP = self.find_best_value(variable=best_var)
            # Best value and the resulting CSP (another copy) in which the best variable (over which we are
            # iterating) is assigned the best value from its domain found by the algorithm.
            if tree_so_far is None:
                tree_so_far = []
            tree_so_far.append(ITERATION)
            print(tree_so_far)
            resulting_CSP.search(tree_so_far)   # Call search on this new copy of self with best variable and value
                                                # assigned.
            tree_so_far.pop()
            if resulting_CSP.CONSISTENT:        # CONSISTENT would either be True or False now.
                self.all_vars = resulting_CSP.all_vars
                return
            else:       # resulting must be False
                best_var.remove_from_domain(best_val)       # If this value was not able to find a CONSISTENT solution
                                                            # Then remove it from the domain.
        return      # If the domain of the variable has been reduced to [].


def main():
    """
    All the universal variables, calls to other relevant functions and print of the output goes in here.
    Variable Description:
        n   --> Number of points.
        d   --> Dimensions of the hypercube.
        t   --> The distance constraint that must be satisfied.
        constrain       --> The constraint class that requires points (rows) as input and returns true of false
                        subject to satisfiability of the constraint.

    """
    n = 16
    d = 7
    t = 3
    Constraint.t = t
    csp1 = CSP(n, d, t)
    # print(list(csp1.all_constraints['point1'].satisfied_through()))
    # csp1.all_constraints['point2'].reduce_domain()
    # csp1.print_status()
    # csp1.arc_consistency()
    csp1.search()
    csp1.print_status()
    pass


if __name__ == '__main__':
    main()
