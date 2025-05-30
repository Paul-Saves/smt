"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

N-dimensional Rosenbrock problem.
"""

import numpy as np

from smt.problems.problem import Problem
from smt.problems.reduced_problem import ReducedProblem
from smt.problems.rosenbrock import Rosenbrock
from smt.utils.options_dictionary import OptionsDictionary


class NdimRosenbrock(Problem):
    @property
    def design_space(self):
        return self._design_space

    @design_space.setter
    def design_space(self, value):
        self._design_space = value

    def __init__(self, ndim=1, w=0.2):
        super().__init__()
        self.problem = ReducedProblem(
            Rosenbrock(ndim=ndim + 1), np.arange(1, ndim + 1), w=w
        )

        self.options = OptionsDictionary()
        self.options.declare("ndim", ndim, types=int)
        self.options.declare("return_complex", False, types=bool)
        self.options.declare("name", "NdimRosenbrock", types=str)

        self.xlimits = self.problem.xlimits

    def _evaluate(self, x, kx):
        return self.problem._evaluate(x, kx)
