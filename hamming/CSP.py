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
        self.CONSISTENT = None      # Set to true if Eqns and Constraints are consistent,
                                    # false if not, and None if unknown
        domain = list(range(2**d))                                      # Domain for each variable
        self.all_vars = [Variable(d, domain) for _ in range(n)]         # List of all Variables by initializing.
        self.all_vars[0].assign_value(0)
        self.all_vars[1].assign_value(7)
        self.all_vars[2].assign_value(7<<4)
        self.all_constraints = dict()                       # Dict to contain all the constraint equations.
        self.constraint_eqn_for_var = [[] for _ in range(len(self.all_vars))]
        # Put the keys of constraint equations dict that are relevant for the variable on the same index.
        self.construct_constraints()

    def print_status(self):
        print("Printing the status of the CSP")
        inconsistencies = [var.inconsistent for var in self.all_vars]
        if any([var.inconsistent for var in self.all_vars]):
            print("The CSP leads to INCONSISTENCY!")
            print(inconsistencies)
            return
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

    def arc_consistency(self):
        """
        A function to check the consistency among all the values in domain of all variables to ensure they are
        consistent i.e. not removable.
        :return: True if arc consistency is maintained. False if domain of a variable is reduced to [].
        """
        ITERATIONS = 0
        FACTOR = 0.8        # The decay factor for the priorities in each loop.
        PRIORITY_VAL = 1    # The added value for the priorities in each loop.
        all_constraints = list(self.all_constraints.keys())                # A list of all constraint to be checked.
        constraints_priority = [PRIORITY_VAL]*2 + [PRIORITY_VAL*FACTOR]*(len(all_constraints)-2)
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
            if any([len(var) == 0 for var in self.all_vars]):
                self.CONSISTENT = False
                return False
        return self.ac_measure()

    def variable_assignment(self, variable=None, value=None):
        """
        This function assigns the most appropriate value to the most appropriate variable pertaining to the
        rules mentioned in the beginning of the code.
        :rtype: (Variable, tuple, CSP)
        :return: None
        """
        variables_to_check = []
        if value is not None and variable is not None:
            variable_idx = self.all_vars.index(variable)
            assert value in variable.domain
            copy_for_assignment = copy.deepcopy(self)
            copy_for_assignment.all_vars[variable_idx].assign_value(value)
            # If the value and variable are known then assign and return.
            return variable, value, copy_for_assignment
        if variable is not None:
            if isinstance(variable, list):  # If only the variable/s is known and it is a list of variables then use
                                            # that list
                for var in variable:
                    assert not var.assigned
                variables_to_check = variable
            else:
                assert not variable.assigned
                variables_to_check = [variable]     # Else put a single element in list to be checked.
        else:
            for var in self.all_vars:
                if not var.assigned:
                    variables_to_check.append(var)
        if value is not None:       # If variable is not provided, then a single value ought to be provided.
            # The algorithm finds all the variables in self that are not already assigned but have value in their domain
            for var in self.all_vars:
                if var.assigned or value not in var.domain:
                    continue
                else:
                    variables_to_check.append(var)
        if len(variables_to_check) < 1:
            raise ValueError
        # Find the best Variable to be assigned a value.
        elements_in_domain = []
        for var in variables_to_check:
            elements_in_domain.append(len(var.domain))
        min_elements_idx = elements_in_domain.index(min(elements_in_domain))
        variable_to_assign = variables_to_check[min_elements_idx]
        # Find the best value to be assigned.
        # best_measure_so_far = float('-inf')
        # best_step_so_far = None
        # best_val_so_far = None
        # for val in variable_to_assign.next_val():
        #     _, _, copy_for_ac = self.variable_assignment(variable_to_assign, val)
        #     ac_measure = copy_for_ac.arc_consistency()
        #     if ac_measure > best_measure_so_far:
        #         best_measure_so_far = ac_measure
        #         best_step_so_far = copy_for_ac
        #         best_val_so_far = val
        best_val_so_far = random.choice(variable_to_assign.domain)
        _, _, best_step_so_far = self.variable_assignment(variable_to_assign, best_val_so_far)
        return variable_to_assign, best_val_so_far, best_step_so_far

    def search(self, tree_so_far = None):
        """
        A search algorithm on the CSP problem posed.
        :return: self with values assigned and CONSISTENT flag set to True if the problem is solved,
        else with CONSISTENT flag set to False.
        """
        copy_for_search = copy.deepcopy(self)
        ITERATION = 0
        while self.CONSISTENT is None:
            ITERATION += 1
            try:
                best_var, best_val, resulting_CSP = copy_for_search.variable_assignment()
            except ValueError:  # If no more value remains to be checked.
                if copy_for_search.arc_consistency() is not False:   # If we are consistent on existing values.
                    self.all_vars = copy_for_search.all_vars
                    self.CONSISTENT = True
                    return
                else:
                    self.CONSISTENT = False
                    return
            # temporarily added when ARC Consistency is not checked in variable assignment.
            if resulting_CSP.arc_consistency() is False:
                self.CONSISTENT = False
                return
            if tree_so_far is None:
                tree_so_far = []
            tree_so_far.append(ITERATION)
            print(tree_so_far)
            resulting_CSP.search(tree_so_far)
            tree_so_far.pop()
            if resulting_CSP.CONSISTENT:
                self.all_vars = resulting_CSP.all_vars
                self.CONSISTENT = True
                return
            else:       # resulting must be False
                if not best_var.remove_from_domain(best_val):   # All values from the domain is exhausted.
                    self.CONSISTENT = False
                    return


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
