import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import scalar_search_wolfe2

    

class LineSearchTool(object):
    
    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if previous_alpha is not None:
            alpha_0 = previous_alpha
        else:
            alpha_0 = 1.0

        # Attempt to use scipy's scalar_search_wolfe2
        try:
            alpha = scalar_search_wolfe2(lambda alpha: oracle.func_directional(x_k, d_k, alpha),
                                          lambda alpha: oracle.grad_directional(x_k, d_k, alpha),
                                          c1=1e-4, c2=0.9)[0]
        except (ValueError, LinAlgError):
            # If scalar_search_wolfe2 fails, fall back to backtracking line search
            alpha = self.backtracking_line_search(oracle, x_k, d_k, alpha_0)

        return alpha

    def backtracking_line_search(self, oracle, x_k, d_k, alpha_0, rho=0.5, c=1e-4):
        alpha = alpha_0
        while oracle.func_directional(x_k, d_k, alpha) > oracle.func(x_k) + c * alpha * np.dot(oracle.grad(x_k), d_k):
            alpha *= rho
        return alpha
    
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = LineSearchTool()
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        start_time = datetime.now()
        grad = oracle.grad(x_k)
        grad_norm = norm(grad)
        if grad_norm < tolerance:
            return x_k, 'success', history

        d_k = -grad
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k)

        if alpha_k is None:
            return x_k, 'computational_error', history

        x_k += alpha_k * d_k

        if trace:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(elapsed_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

    return x_k, 'success', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        start_time = datetime.now()

        # Compute gradient and Hessian at current point x_k
        grad_k = oracle.grad(x_k)
        hess_k = oracle.hess(x_k)

        # Check if the Hessian is positive definite
        try:
            cholesky_factor, _ = cho_factor(hess_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        # Solve the linear system Hessian * d_k = -gradient to find Newton direction
        newton_direction = cho_solve((cholesky_factor, True), -grad_k)

        # Perform line search to find step size
        alpha_k = line_search_tool.line_search(oracle, x_k, newton_direction)

        if alpha_k is None:
            return x_k, 'computational_error', history

        # Update the current point
        x_k = x_k + alpha_k * newton_direction

        # Compute function value and gradient norm at the new point
        func_k = oracle.func(x_k)
        grad_norm_k = np.linalg.norm(oracle.grad(x_k))

        # Update history if tracing is enabled
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(func_k)
            history['grad_norm'].append(grad_norm_k)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        # Check for convergence
        if grad_norm_k < tolerance:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history
