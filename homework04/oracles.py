import numpy as np
import scipy
from scipy.special import expit



class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 

class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        m = len(self.b)
        logits = -self.b * self.matvec_Ax(x)
        log_term = np.logaddexp(0, logits)
        reg_term = 0.5 * self.regcoef * np.linalg.norm(x) ** 2
        return (1 / m) * np.sum(log_term) + reg_term

    def grad(self, x):
        m = len(self.b)
        logits = -self.b * self.matvec_Ax(x)
        sigmoid_probs = expit(logits)
        grad_log_term = -self.matvec_ATx(sigmoid_probs) @ self.b
        reg_term = self.regcoef * x
        return (1 / m) * grad_log_term + reg_term

    def hess(self, x):
        m = len(self.b)
        logits = -self.b * self.matvec_Ax(x)
        sigmoid_probs = expit(logits)
        diag_H = self.b ** 2 * (sigmoid_probs * (1 - sigmoid_probs))
        hess_log_term = self.matmat_ATsA(diag_H)
        reg_term = self.regcoef * np.eye(len(x))
        return (1 / m) * hess_log_term + reg_term


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.Ax = matvec_Ax(np.zeros_like(b))  # Precompute Ax for efficiency
        self.matvec_ATx_d = lambda d: matvec_ATx(self.Ax * d)

    def func_directional(self, x, d, alpha):
        Ax_d = self.matvec_Ax(alpha * d)  # Compute Ax for the directional vector
        return np.mean(np.logaddexp(0, -self.b * (Ax_d + self.Ax))) + 0.5 * self.regcoef * np.linalg.norm(x + alpha * d) ** 2

    def grad_directional(self, x, d, alpha):
        Ax_d = self.matvec_Ax(alpha * d)  # Compute Ax for the directional vector
        grad = -self.b * (self.matvec_ATx_d(Ax_d) + self.Ax)
        grad += self.regcoef * (x + alpha * d)  # Regularization term
        return grad




def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):

    matvec_Ax = lambda x: matvec_Ax(x)
    matvec_ATx = lambda x: matvec_ATx(x)

    def matmat_ATsA(s):
        return matmat_ATsA(s)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type: {}'.format(oracle_type))
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    num_features = len(x)
    grad_approx = np.zeros(num_features)
    for i in range(num_features):
        e_i = np.zeros(num_features)
        e_i[i] = 1
        grad_approx[i] = (func(x + eps * e_i) - func(x)) / eps
    return grad_approx

def hess_finite_diff(func, x, eps=1e-5):
    num_features = len(x)
    hess_approx = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            e_i = np.zeros(num_features)
            e_j = np.zeros(num_features)
            e_i[i] = 1
            e_j[j] = 1
            hess_approx[i][j] = (func(x + eps * e_i + eps * e_j) - func(x + eps * e_i) - func(x + eps * e_j) + func(x)) / eps**2
    return hess_approx
