from rem.utils import *

from linear_operator.operators import KroneckerProductLinearOperator, IdentityLinearOperator, DenseLinearOperator, CatLinearOperator, ConstantMulLinearOperator, SumLinearOperator
from linear_operator.utils import stable_pinverse

import torch
import numpy as np

from tqdm import tqdm

class Workload:
    def __init__(self, matrices):
        """
        Instantiates workload for kronecker product of matrices
        :param matrices: list of linear operators
        """
        self.matrices = matrices
        self.operator = KroneckerProductLinearOperator(*self.matrices)
        self.shape = self.operator.shape
        self.zero = False
        for i, opr in enumerate(self.matrices):
            mat = opr.to_dense()
            if mat.max().item() == 0:
                if mat.min().item() == 0:
                    self.zero = True
                    break

    def __matmul__(self, other):
        if isinstance(other, Workload):
            assert len(self.matrices) == len(other.matrices), '# of matrices mismatch'
            assert self.shape[1] == other.shape[0], 'Dimension mismatch'
            assert all([(self.matrices[idx].shape[1] == other.matrices[idx].shape[0]) for idx in range(len(self.matrices))])
            return Workload([DenseLinearOperator(self.matrices[idx].to_dense() @ other.matrices[idx].to_dense()) for idx in range(len(self.matrices))])
        if (isinstance(other, torch.Tensor) and other.dim() == 1):
            if self.zero:
                return torch.zeros(self.shape[0])
            else:
                return self.operator @ other
        if isinstance(other, HStack):
            return HStack([self @ wkload for wkload in other.workloads])
        else:
            NotImplemented
        
    def getAnswers(self, dataset, sigma, return_sigma = False):
        dvec = torch.tensor(dataset.datavector(), dtype = torch.float32)
        sensitivity = torch.sum(self.operator, 0).max().item()
        noise = sensitivity * sigma * torch.randn(self.shape[0])
        answers = self.operator @ dvec + noise
        if return_sigma == True:
            return (answers, sigma)
        else:
            return answers
        
    def toDense(self):
        return self.operator.to_dense()
    
    def T(self):
        return Workload([mat.T for mat in self.matrices ])
    
    def pinv(self):
        return Workload([DenseLinearOperator(stable_pinverse(mat.to_dense())) for mat in self.matrices])
    
class MarginalWorkload(Workload):
    def __init__(self, cols, domain):
        """
        Instantiates marginal workload for cols
        :param cols: set of column names
        :param domain: Domain object
        """
        self.cols = tuple(cols)
        self.domain = domain
        self.matrices = [IdentityLinearOperator(self.domain[col]) if col in self.cols else _construct_total_query(self.domain[col]) for col in self.domain.attrs]
        self.operator = KroneckerProductLinearOperator(*self.matrices)
        self.shape = self.operator.shape
        
    def getAnswers(self, dataset, sigma = None, rho = None, return_sigma = False):
        assert ((sigma is None) and rho is not None) or ((rho is None) and sigma is not None), 'Exactly one of sigma and rho must be defined'
        
        if rho:
            sigma = ((2 * rho) ** -0.5)
        if self.cols == ():
            marg_answers = torch.tensor(dataset.df.shape[0], dtype = torch.float32)
            answers = marg_answers + sigma * torch.randn(1)
            if return_sigma == True:
                return (answers, sigma)
            else:
                return answers  
        else:
            marg_answers = torch.tensor(dataset.project(self.cols).datavector(), dtype = torch.float32)
            noise = sigma * torch.randn(len(marg_answers))
            answers = marg_answers + noise
            if return_sigma == True:
                return (answers, sigma)
            else:
                return answers 
       
    def __matmul__(self, other):
        if isinstance(other, Workload):
            assert len(self.matrices) == len(other.matrices), '# of matrices mismatch'
            assert self.shape[1] == other.shape[0], 'Dimension mismatch'
            assert all([(self.matrices[idx].shape[1] == other.matrices[idx].shape[0]) for idx in range(len(self.matrices))])
            if isinstance(other, ResidualWorkload2Pinv):
                return Workload([matmulMarRes2Pinv(self.matrices[idx], other.matrices[idx]) for idx in range(len(self.matrices))])
            else:
                return Workload([DenseLinearOperator(self.matrices[idx].to_dense() @ other.matrices[idx].to_dense()) for idx in range(len(self.matrices))])
        if (isinstance(other, torch.Tensor) and other.dim() == 1):
            return self.operator @ other
        if isinstance(other, HStack):
            return HStack([self @ wkload for wkload in other.workloads])
        else:
            NotImplemented
            
    def decomposeIntoResiduals(self, dataset = None, y = None, sigma = None, rho = None):
        residuals = VStack([ResidualWorkload2(tup, self.domain) for tup in downward_closure([self.cols])])
        if isinstance(y, torch.Tensor):
            assert isinstance(sigma, float) or isinstance(sigma, int), 'sigma not defined'
            ym = y
        else:
            ym, sigma = self.getAnswers(dataset, sigma = sigma, rho = rho, return_sigma = True)
        yr = []
        sigmas = []
        M_plus = self.pinv()
        for wkload in residuals.workloads:
            yr.append((wkload @ M_plus) @ ym )
            sigmas.append(sigma * np.prod([ self.domain[i] for i in self.cols if i not in wkload.cols ]) ** 0.5)
        return (residuals, yr, sigmas)
        
class ResidualWorkload(Workload):
    def __init__(self, cols, domain):
        """
        Instantiates residual workload for cols using version from ResidualPlanner paper (https://arxiv.org/abs/2305.08175)
        :param cols: set of column names
        :param domain: Domain object
        """
        self.cols = tuple(cols)
        self.domain = domain
        self.matrices = [_construct_sub_matrix(self.domain[col]) if col in self.cols else _construct_total_query(self.domain[col]) for col in self.domain.attrs]
        self.operator = KroneckerProductLinearOperator(*self.matrices)
        self.shape = self.operator.shape
        
    def getAnswers(self, dataset, sigma = None, rho = None, return_sigma = False, isotrophic = False):
        assert ((sigma is None) and rho is not None) or ((rho is None) and sigma is not None), 'Exactly one of sigma and rho must be defined'
        if isotrophic:
            if rho:
                sigma = ((2 * rho) ** -0.5)
            if self.cols == ():
                marg_answers = torch.tensor(dataset.df.shape[0], dtype = torch.float32)
                res_answers = marg_answers + sigma * torch.randn(1)
                if return_sigma == True:
                    return (res_answers, sigma)
                else:
                    return res_answers 
            else:
                marg_answers = torch.tensor(dataset.project(self.cols).datavector(), dtype = torch.float32)
                H = KroneckerProductLinearOperator(*[_construct_sub_matrix(self.domain[col]) for col in self.cols])
                res_answers =  H @ marg_answers
                sensitivity = torch.sum(H, 0).max() ** 0.5
                noise = sensitivity * sigma * torch.randn(len(res_answers))
                if return_sigma == True:
                    return (res_answers + noise, sigma)
                else:
                    return res_answers + noise     
            
        else:
            if rho:
                sigma = ((1/(2*rho)) * np.prod([(self.domain[col] - 1)/self.domain[col] for col in self.cols]) ) ** 0.5

            if self.cols == ():
                marg_answers = torch.tensor(dataset.df.shape[0], dtype = torch.float32)
                res_answers = marg_answers + sigma * torch.randn(1)
                if return_sigma == True:
                    return (res_answers, sigma)
                else:
                    return res_answers 
            else:
                marg_answers = torch.tensor(dataset.project(self.cols).datavector(), dtype = torch.float32)
                H = KroneckerProductLinearOperator(*[_construct_sub_matrix(self.domain[col]) for col in self.cols])
                noise = sigma * torch.randn(len(marg_answers))
                res_answers =  H @ (marg_answers + noise)
                if return_sigma == True:
                    return (res_answers, sigma)
                else:
                    return res_answers 
    
    def pinv(self):
        return Workload([_construct_sub_matrix_pinverse(self.domain[col]) if col in self.cols else _construct_total_query_pinv(self.domain[col]) for col in self.domain.attrs])

class ResidualWorkload2(Workload):
    def __init__(self, cols, domain):
        """
        Instantiates residual workload for cols using version from ReM paper (https://arxiv.org/abs/2410.01091)
        :param cols: set of column names
        :param domain: Domain object
        """
        self.cols = tuple(cols)
        self.domain = domain
        self.matrices = [_construct_contrast_basis(self.domain[col]) if col in self.cols else _construct_total_query(self.domain[col]) for col in self.domain.attrs]
        self.operator = KroneckerProductLinearOperator(*self.matrices)
        self.shape = self.operator.shape
        
    def getAnswers(self, dataset, sigma = None, rho = None, return_sigma = False):
        assert ((sigma is None) and rho is not None) or ((rho is None) and sigma is not None), 'Exactly one of sigma and rho must be defined'
        
        if rho:
            sigma = ((1/(2*rho)) * np.prod([(self.domain[col] - 1)/self.domain[col] for col in self.cols]) ) ** 0.5
        
        if self.cols == ():
            marg_answers = torch.tensor(dataset.df.shape[0], dtype = torch.float32)
            res_answers = marg_answers + sigma * torch.randn(1)
            if return_sigma == True:
                return (res_answers, sigma)
            else:
                return res_answers 
        else:
            marg_answers = torch.tensor(dataset.project(self.cols).datavector(), dtype = torch.float32)
            H = KroneckerProductLinearOperator(*[_construct_contrast_basis(self.domain[col]) for col in self.cols])
            noise = sigma * torch.randn(len(marg_answers))
            res_answers =  H @ (marg_answers + noise)
            if return_sigma == True:
                return (res_answers, sigma)
            else:
                return res_answers 
    
    def pinv(self):
        return ResidualWorkload2Pinv(self.cols, self.domain)

    def __matmul__(self, other):
        if isinstance(other, Workload):
            assert len(self.matrices) == len(other.matrices), '# of matrices mismatch'
            assert self.shape[1] == other.shape[0], 'Dimension mismatch'
            assert all([(self.matrices[idx].shape[1] == other.matrices[idx].shape[0]) for idx in range(len(self.matrices))])
            if isinstance(other, ResidualWorkload2Pinv):
                return Workload([matmulRes2Res2Pinv(self.matrices[idx], other.matrices[idx]) for idx in range(len(self.matrices))])
            else:
                return Workload([DenseLinearOperator(self.matrices[idx].to_dense() @ other.matrices[idx].to_dense()) for idx in range(len(self.matrices))])
        if (isinstance(other, torch.Tensor) and other.dim() == 1):
            return self.operator @ other
        if isinstance(other, HStack):
            return HStack([self @ wkload for wkload in other.workloads])
        else:
            NotImplemented
    
class ResidualWorkload2Pinv(Workload):
    def __init__(self, cols, domain):
        """
        Instantiates pseudoinverse of residual workload for cols
        :param cols: set of column names
        :param domain: Domain object
        """
        self.cols = tuple(cols)
        self.domain = domain
        self.matrices = [_construct_contrast_basis_pinv(self.domain[col]) if col in self.cols else _construct_total_query_pinv(self.domain[col]) for col in self.domain.attrs]
        self.operator = KroneckerProductLinearOperator(*self.matrices)
        self.shape = self.operator.shape
        self.zero = False

    def pinv(self):
        return ResidualWorkload2(self.cols, self.domain)
        
class RangeWorkload(Workload):
    def __init__(self, cols, domain):
        """
        Instantiates range workload for kronecker product of matrices
        :param cols: list of columns
        :param domain: data domain dictionary
        """
        self.cols = tuple(cols)
        self.domain = domain
        self.matrices = [_construct_range_matrix(self.domain[col]) if col in self.cols else _construct_total_query(self.domain[col]) for col in self.domain.attrs]
        self.operator = KroneckerProductLinearOperator(*self.matrices)
        self.shape = self.operator.shape
        
    def getAnswers(self, dataset, sigma = None, rho = None, return_sigma = False):
        assert ((sigma is None) and rho is not None) or ((rho is None) and sigma is not None), 'Exactly one of sigma and rho must be defined'
        
        if rho:
            sigma = ((2 * rho) ** -0.5)
        
        marg_answers = torch.tensor(dataset.project(self.cols).datavector(), dtype = torch.float32)
        range_answers = KroneckerProductLinearOperator(*[self.matrices[int(col)] for i, col in enumerate(self.cols)]) @ marg_answers
        sensitivity = np.prod([torch.sum(mat, 0).max().item() for mat in self.matrices]) ** 0.5
        noise = sigma * sensitivity * torch.randn(len(range_answers))
        answers = range_answers + noise
        if return_sigma == True:
            return (answers, sigma)
        else:
            return answers 
        
class VStack:
    def __init__(self, workloads):
        """
        Instantiates vertical stack of workloads
        :param  workloads: list of Workloads or ResidualWorkloads
        """
        self.workloads = workloads
        
        # check that horizontal shape of workloads match
        if len(set([wkload.shape[1] for wkload in self.workloads])) != 1:
            raise Exception('Workload dimensions do not match')
        
        self.shape = [np.sum([wkload.shape[0] for wkload in self.workloads]), self.workloads[0].shape[1]]

    def __matmul__(self, other):
        if isinstance(other, HStack):
            return Block([wkload @ other for wkload in self.workloads])
            
        if (isinstance(other, torch.Tensor) and other.dim() == 1):
            return [wkload @ other for wkload in self.workloads]
        
    def __add__(self, other):
        if isinstance(other, VStack):
            return VStack(self.workloads + other.workloads)
        else:
            NotImplemented
            
    def cols(self):
        try:
            wkload_cols = [wkload.cols for wkload in self.workloads]
        except:
            raise Exception('Not all workloads have cols method')
        else:
            return wkload_cols
        
    def getAnswers(self, dataset, sigma = None, rho = None, return_sigma = False):
        """
        Instantiates vertical stack of workloads
        :param sigma: scalar or list of scalars
        :param rho: scalar or list of scalars
        """
        assert ((sigma is None) and rho is not None) or ((rho is None) and sigma is not None), 'Exactly one of sigma and rho must be defined'
        
        if sigma is not None:
            if isinstance(sigma, list):
                assert len(sigma) == len(self.workloads), 'sigma not correct length'
                wkload_answers = [self.workloads[idx].getAnswers(dataset, sigma = sigma[idx], return_sigma = return_sigma) for idx in range(len(self.workloads))]

            else:
                wkload_answers = [wkload.getAnswers(dataset, sigma = sigma, return_sigma = return_sigma) for wkload in self.workloads]
        
        if rho is not None:
            if isinstance(rho, list):
                assert len(rho) == len(self.workloads), 'rho not correct length'
                wkload_answers = [self.workloads[idx].getAnswers(dataset, rho = rho[idx], return_sigma = return_sigma) for idx in range(len(self.workloads))]

            else:
                wkload_answers = [wkload.getAnswers(dataset, rho = rho, return_sigma = return_sigma) for wkload in self.workloads]
        
        if (return_sigma == True):
            return [list(out) for out in zip(*wkload_answers)]
        else:
            return wkload_answers
           
    def pinv(self):
        if all([isinstance(wkload, ResidualWorkload) for wkload in self.workloads]) or all([isinstance(wkload, ResidualWorkload2) for wkload in self.workloads]):
            return HStack([wkload.pinv() for wkload in self.workloads])
        else:
            raise Exception('pinv only valid if all blocks are ResidualWorkloads')
            
    def append(self, wkload):
        assert self.workloads[0].shape[1] == wkload.shape[1], 'appended workload dimensions do not match existing workloads'
        return VStack(self.workloads + [wkload])
    
    def toDense(self):
        return torch.concat([wkload.toDense() for wkload in self.workloads], axis = 0)
    
    def subStack(self, indices):
        return VStack([self.workloads[i] for i in indices])
    
    def decomposeIntoResiduals(self, dataset = None, y = None, sigma = None, rho = None):
        assert all([isinstance(wkload, MarginalWorkload) for wkload in self.workloads]), 'All workloads must be MarginalWorkloads'
        for i, mar in enumerate(self.workloads):
            if y:
                y_input = y[i]
            else:
                y_input = None
            if sigma:
                sigma_input = sigma[i]
            else:
                sigma_input = None
            if rho: 
                rho_input = rho[i]
            else:
                rho_input = None
            current_res, current_ans, current_sig = mar.decomposeIntoResiduals(dataset = dataset, 
                                                                               y = y_input, 
                                                                               sigma = sigma_input, 
                                                                               rho = rho_input)
            if i == 0:
                residuals = current_res
                answers = current_ans
                sigmas = current_sig
            else:
                residuals += current_res
                answers += current_ans
                sigmas += current_sig
        return (residuals, answers, sigmas)
              
class HStack:
    def __init__(self, workloads):
        """
        Instantiates horizontal stack of workloads
        :param matrices: list of Workloads or ResidualWorkloads
        """
        self.workloads = workloads

        # check that vertical shape of workloads match
        if len(set([wkload.shape[0] for wkload in self.workloads])) != 1:
            raise Exception('Workload dimensions do not match')

        self.shape = [self.workloads[0].shape[0], np.sum([wkload.shape[1] for wkload in self.workloads])]

    def __matmul__(self, other):
        if isinstance(other, list):
            # check list length matches # of workloads
            assert len(other) == len(self.workloads), 'List length does not match workload length'
            # check that length of list elements matches width of workloads
            assert all([len(other[idx]) == self.workloads[idx].shape[1] for idx in range(len(other))]), 'Some vector does not match workload dimension'
            
            return torch.stack([self.workloads[idx] @ other[idx] for idx in range(len(other))]).sum(axis = 0)

class Block:
    def __init__(self, stacks):
        """
        Instantiates a block matrix (vertical stack of HStacks)
        :param stacks: list of HStacks
        """
        # check that all inputs are HStacks
        assert all([isinstance(stk, HStack) for stk in stacks]), 'Not all inputs are HStacks'
        # check that all inputs have same width
        assert len(set([stk.shape[1] for stk in stacks])) == 1, 'Dimension mismatch'
        # check that all inputs have same # of HStacks
        assert len(set([len(stk.workloads) for stk in stacks])) == 1, '# of HStacks mismatch'
        
        self.stacks = stacks 
        
    def __matmul__(self, other):
        if isinstance(other, list):
            return [stk @ other for stk in tqdm(self.stacks)]
    
def _construct_sub_matrix(size):
    """
    Construct the subtraction matrix corresponding to an attribute with att_size values, returned linear operator has shape (att_size - 1, att_size)

    Arguments:
    att_size: the number of values the attribute can take on (positive integer)

    Returns:
    the subtraction matrix (linear operator)
    """
    if size < 2:
        raise Exception('Attribute size must be at least 2')
    n = size - 1
    ones = DenseLinearOperator(torch.ones(n, 1))
    negative_identity = -1 * IdentityLinearOperator(n)
    return CatLinearOperator(ones, negative_identity, dim = 1)

def _construct_sub_matrix_pinverse(size):
    """
    Construct the pseudo-inverse of the subtraction matrix corresponding to an attribute with att_size values, returned linear operator has shape (att_size, att_size - 1)

    Arguments:
    att_size: the number of values the attribute can take on (positive integer)

    Returns:
    the pseudo-inverse of the subtraction matrix (linear operator)
    """
    if size < 2:
        raise Exception('Attribute size must be at least 2')
    n = size - 1
    ones = DenseLinearOperator(torch.ones(1, n))
    off_row = DenseLinearOperator(torch.ones(n, n)) + (IdentityLinearOperator(n) * (-1 * size))
    return (1 / size) * CatLinearOperator(ones, off_row, dim = 0)

def _construct_total_query(size):
    return DenseLinearOperator(torch.ones(1, size))

def _construct_total_query_pinv(size):
    return DenseLinearOperator(torch.ones(size, 1)) * (1 / size)

def _construct_contrast_basis(size):
    
    if size < 2:
        raise Exception('Attribute size must be at least 2')
    n = size - 1
    zeros = DenseLinearOperator(torch.zeros(n, 1))
    identity = IdentityLinearOperator(n)
    L = CatLinearOperator(identity, zeros, dim = 1)
    R = CatLinearOperator(zeros, identity, dim = 1)
    return L - R

def _construct_contrast_basis_pinv_bruteforce(size):
    if size < 2:
        raise Exception('Attribute size must be at least 2')
    n = size - 1
    identity = torch.eye(n)
    zeros = torch.zeros(n, 1)
    L = torch.concat([identity, zeros], dim = 1)
    R = torch.concat([zeros, identity], dim = 1)    
    return DenseLinearOperator(stable_pinverse(L - R))

def _construct_contrast_basis_pinv(size):
    if size < 2:
        raise Exception('Attribute size must be at least 2')
    n = size
    tri = torch.triu(torch.ones([n, n - 1]))
    ones = torch.ones([n, n - 1])
    diag = torch.diag(torch.arange(1, n, dtype = torch.float32))
    return DenseLinearOperator(tri - 1/n * (ones @ diag) )

def matmulMarRes2Pinv(left, right):
    if isinstance(left, IdentityLinearOperator):
        return right
    elif isinstance(left, DenseLinearOperator):
        # Total X Total^+ Case
        if isinstance(right, ConstantMulLinearOperator):
            return IdentityLinearOperator(1)
        # Total X Contrast^+ Case
        elif isinstance(right, DenseLinearOperator):
            return DenseLinearOperator(torch.zeros(1, right.shape[1]))
        else:
            raise Exception('Workload type mismatch')
    else:
        raise Exception('Workload type mismatch')
        
def matmulRes2Res2Pinv(left, right):
    if isinstance(left, SumLinearOperator):
        # Contrast X Total^+ Case
        if isinstance(right, ConstantMulLinearOperator):
            return DenseLinearOperator(torch.zeros(left.shape[0], 1))
        # Contrast X Contrast^+ Case
        elif isinstance(right, DenseLinearOperator):
            return IdentityLinearOperator(left.shape[0])
        else:
            raise Exception('Workload type mismatch')
    elif isinstance(left, DenseLinearOperator):
        # Total X Total^+ Case
        if isinstance(right, ConstantMulLinearOperator):
            return IdentityLinearOperator(1)
        # Total X Contrast^+ Case
        elif isinstance(right, DenseLinearOperator):
            return DenseLinearOperator(torch.zeros(1, right.shape[1]))
        else:
            raise Exception('Workload type mismatch')
    else:
        raise Exception('Workload type mismatch')
        
def _construct_range_matrix(size):
    if size < 2:
        raise Exception('Attribute size must be at least 2')
    n = size
    mat = torch.concat([torch.concat([ torch.zeros(n - k, k), torch.tril(torch.ones([n - k, n - k]))], axis = 1) for k in range(n)], axis = 0)
    return DenseLinearOperator(mat)