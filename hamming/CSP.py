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

class Variable():
    """
    This class handles all the function sand state of a single variable.
    """

    def __init__(self, d, domain = None):
        """
        :param d: Dimensions of the Variable
        :param domain: Initial domain. If not provided, use range(2**d) as the domain.
        """
        self.d = d
        self.__domain = copy.deepcopy(domain) if domain is not None else list(range(2**d))
        self._enum_domain = []
        self.reset()

    def list_to_dec(self, list):
        """
        Convert a list of boolean into its equivalent decimal integer.
        """
        powered = [elem*2**(self.d-idx-1) for idx, elem in enumerate(list)]
        return sum(powered)

    def dec_to_list(self, val):
        """
        Convert a decimal number into its equivalent list of booleans.
        """
        return [(val>>(i-1))%2 for i in range(self.d,0,-1)]

    def reset(self, expand_domain = False):
        """
        Reset the object and set the domain to initial domain.
        """
        self.assigned = False
        self._enum_domain = []
        if expand_domain:
            self.__domain = list(range(2**self.d))
        for val in self.__domain:
            self.add_to_domain(val)

    def add_to_domain(self, val):
        """
        Add a value to domain.
        """
        if isinstance(val, list):
            val = self.list_to_dec(val)
        if val not in self._enum_domain:
            self._enum_domain.append(val)

    def remove_from_domain(self, val):
        """
        Remove the provided value from domain enum.
        :return: Raise ValueType Error if the val is not present in the domain.
        Returns False if the domain enum becomes empty post the removal of the val.
        Returns True otherwise.
        """
        if isinstance(val, list):
            val = self.list_to_dec(val)
        try:
            self._enum_domain.remove(val)
        except ValueError:
            raise
        if len(self._enum_domain) == 0:
            return False
        return True

    def all_vals(self):
        """
        :return: A list of all values in the domain.
        """
        return [self.dec_to_list(val) for val in self._enum_domain]

    def next_val(self):
        """
        An iteration over the values in domain.
        """
        for val in self._enum_domain:
            yield self.dec_to_list(val)

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
        if isinstance(val, list):
            val = self.list_to_dec(val)
        self._enum_domain = [val]

    def __len__(self):
        """
        :return: THe size of domain of variable
        """
        return len(self._enum_domain)

class Constraint():
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
        :return: An iter over all the tuple of values from relevant variables that satisfy the constraint.
        """
        domains = [var.all_vals() for var in self.vars]
        value_iter = product(*domains)
        for values in value_iter:
            if self.constraint_function(*values, *self.args, **self.kwargs):
                # yield values if len(values)>1 else yield values[0]
                yield values

    def reduced_domain(self):
        all_possible_vals = list(self.satisfied_through())      # List of all entries (in form of list) from var1, var2,
                                        # var3, ... that satisfies the constraint.
        list_of_possible_vals = list(map(list, zip(*all_possible_vals)))
                                        # [List of permissible values in var1, List of permissible values in var2, ...]
        for vals, var in zip(list_of_possible_vals, self.vars):     # For each var, find the set of unique vals in its
            # domain subject to the constraints and change the domain to the same.
            var.assign_domain(list(vals))       # The domain handling in variable takes care than the same element
            # is not repeated in the domain.


    @classmethod
    def no_conflict(cls, val1, val2):
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
        return sum(conflict)<2

    @classmethod
    def less_than_t_conflicts(cls, t, var1, var2 = None):
        if var2 is None:
            # Variable1 is a matrix realization and we have to check for consistency among all rows.
            for idx1, row1 in enumerate(var1):
                for row2 in var1[idx1+1:]:
                    if not cls.no_conflict(row1, row2):
                        return False
            return True

        if isinstance(var2, int):
            # Variable1 is a matrix realization and we have to check consistency for the var2'th row value assignment
            # only.
            for idx, row in enumerate(var1):
                if idx == var2:
                    continue
                if not cls.no_conflict(var1[var2], row):
                    return False
            return True

        # Assume var1 and var2 to be two values assigned to points and check the conflict
        return cls.no_conflict(var1, var2)

class CSP():
    """
    A class to maintain the Problem itself.
    """
    def __init__(self, n, d, t):
        self.domain = list(range(2**d))
        self.all_vars = [Variable(d, self.domain) for i in range(n)]
        self.all_constraints = dict()
        for idx1, var1 in enumerate(self.all_vars):
            for idx2, var2 in enumerate(self.all_vars):
                if idx2 > idx1:
                    self.all_constraints[(idx1, idx2)] = Constraint(Constraint.no_conflict, 2, var1, var2)
        self.all_constraints['point1'] = Constraint(Constraint.equal, 1, self.all_vars[0], [0]*d)
        self.all_constraints['point2'] = Constraint(Constraint.equal, 1, self.all_vars[1], [0]*(d-t)+[1]*t)



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
    print(list(csp1.all_constraints['point1'].satisfied_through()))
    csp1.all_constraints['point2'].reduced_domain()
    pass

if __name__ == '__main__':
    main()