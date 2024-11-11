from rem.algebra import VStack
from rem.dual_ascent import dualAscent
import torch

def ivwFromResiduals(M, y, sigmas):
    M_gen = []
    y_gen = []
    sigmas_gen = []
    T_M = M.cols()
    for tup in set(T_M):
        first_idx = T_M.index(tup)
        M_gen.append(M.workloads[first_idx])
        numerator = sum([y[idx] / sigmas[idx] ** 2 for idx, cols in enumerate(T_M) if cols == tup])
        denominator = sum([sigmas[idx] ** -2 for idx, cols in enumerate(T_M) if cols == tup])
        y_gen.append(numerator/denominator)
        sigmas_gen.append(denominator ** -0.5)
    M_gen = VStack(M_gen)
    return M_gen, y_gen, sigmas_gen

class gremMLE:
    def __init__(self, target_marginals, residuals, res_answers, res_sigmas):
        self.target_marginals = target_marginals
        self.residuals = residuals
        self.res_answers = res_answers
        self.res_sigmas = res_sigmas
        self.R, self.z, self.z_sigmas = ivwFromResiduals(self.residuals, self.res_answers, self.res_sigmas)
        self.target_marg_answers = (self.target_marginals @ self.R.pinv()) @ self.z

    def getMarginals(self, postprocessing = None):
        if postprocessing == 'trunc':
            return [torch.max(y_gamma, torch.tensor(0)) for i, y_gamma in enumerate(self.target_marg_answers)]
        elif postprocessing == 'trunc+rescale':
            target_marg_answers_trunc = [torch.max(y_gamma, torch.tensor(0)) for i, y_gamma in enumerate(self.target_marg_answers)]
            return [target_marg_answers_trunc[i] * self.target_marg_answers[i].sum() / target_marg_answers_trunc[i].sum() for i, _ in enumerate(self.target_marg_answers)]
        else:
            return self.target_marg_answers

class gremLNN:
    def __init__(self, target_marginals, residuals, res_answers, res_sigmas):
        self.target_marginals = target_marginals
        self.residuals = residuals
        self.res_answers = res_answers
        self.res_sigmas = res_sigmas
        self.domain = self.residuals.workloads[0].domain
        self.optimizer = dualAscent(T_M = residuals.cols(), 
                                    T_Q = self.target_marginals.cols(), 
                                    y = res_answers, 
                                    sigmas =  [2 ** len(tup) for tup in residuals.cols()], 
                                    domain = self.domain)
    
    def solve(self, lam = -0.1, t = 1, t_div = 10, early_stopping = 0.01, reg_param = 1):
        self.optimizer.solveLooping(rounds = 4001, lam = lam, t = t, t_div = t_div, early_stopping = early_stopping, reg_param = reg_param)
        self.running_time = self.optimizer.running_time

    def getMarginals(self):
        self.z = [self.optimizer.y_opt_dict[tup] for tup in self.optimizer.R_Q]
        self.target_marg_answers = (self.target_marginals @ self.optimizer.residuals_all.pinv()) @ self.z
        return self.target_marg_answers

class emp:
    def __init__(self, target_marginals, marginals, mar_answers, mar_sigmas):
        self.target_marginals = target_marginals
        self.marginals = marginals
        self.mar_answers = mar_answers
        self.mar_sigmas = mar_sigmas
        self.domain = self.marginals.workloads[0].domain
        self.R, self.z, self.z_sigmas = marginals.decomposeIntoResiduals(y = self.mar_answers, sigma = self.mar_sigmas)

    def getMarginals(self):
        self.inferred = gremMLE(self.marginals, self.R, self.z, self.z_sigmas)
        return self.inferred.getMarginals()