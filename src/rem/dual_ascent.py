from rem.utils import downward_closure
from rem.algebra import VStack, MarginalWorkload, ResidualWorkload2, _construct_contrast_basis, Workload
import numpy as np
import torch
import time
import math

def getResidualLength(tup, domain):
    return np.prod([domain[col] - 1 for col in tup])

def getMarginalLength(tup, domain):
    return np.prod([domain[col] for col in tup])

def getCovarianceMat(tup, domain):
    if tup == ():
        return torch.tensor([1,])
    else:
        H = Workload([_construct_contrast_basis(domain[col]) for col in tup])
        return H @ H.T()
    
class dualAscent:
    def __init__(self, T_M, T_Q, y, sigmas, domain):
        self.domain = domain
        self.T_M = T_M
        self.residuals = VStack([ResidualWorkload2(tup, self.domain) for tup in T_M])
        self.T_Q = T_Q
        self.marginals = VStack([MarginalWorkload(tup, self.domain) for tup in self.T_Q])
        self.R_Q = downward_closure(T_Q)
        self.residuals_all = VStack([ResidualWorkload2(tup, self.domain) for tup in self.R_Q])
        self.y = y
        self.sigmas = sigmas
        
    def solve(self, rounds = 2001, lam = -0.1, t = 0.01, early_stopping = 0.01, true_answers = None, num_records = None):
        self.rounds = rounds
        self.t = t
        self.covar = [(self.sigmas[idx] ** 2, getCovarianceMat(tup, self.domain)) for idx, tup in enumerate(self.T_M)]
        marg_stats = []
        marg_nnsum = []
        obj_stats = []
        obj_main_stats = []
        error_stats = []
        dual_gap_stats = []
        total_stats = []
        start_time = time.time()
        
        lam_dict = { tup : lam * torch.ones((getMarginalLength(tup, self.domain),)) for tup in self.T_Q }
        
        
        inv_sum_inv_sigma = { tup : (1/(np.sum([(1 / self.covar[idx][0]) for idx, cols in enumerate(self.T_M) if cols == tup])) 
                                     if tup in self.T_M else 1)
                     for tup in self.R_Q 
                    }
        
        # sum_inv_matvec (handle total query separately)
        sum_inv_matvec = { tup : (torch.stack([ (1/self.covar[idx][0]) * (self.covar[idx][1].pinv() @ self.y[idx]) 
                                               for idx, cols in enumerate(self.T_M) if cols == tup]).sum(dim = 0) if tup in self.T_M
                          else torch.zeros((getResidualLength(tup, self.domain),)))
                          for tup in self.R_Q 
                          if tup != ()
                         }
        sum_inv_matvec[()] = torch.stack( [ self.y[idx] * (1/self.covar[idx][0]) for idx, cols in enumerate(self.T_M) if cols == ()]).sum(dim = 0)
        
        for i in range(rounds):
            # update lambda map
            lambda_map = { tup : torch.stack([ ( (self.marginals.workloads[self.T_Q.index(cols)] @ self.residuals_all.workloads[self.R_Q.index(tup)].pinv() ).T() @ lam_dict[cols] )
                                              for cols in self.T_Q if set(tup).issubset(set(cols))]).sum(dim = 0)
                          for tup in self.R_Q
                         }
            # calc new alpha star
            alpha_star = { 
                tup : 
                (inv_sum_inv_sigma[tup] * (self.covar[self.T_M.index(tup)][1] @ (sum_inv_matvec[tup] - lambda_map[tup]))
                      if tup in self.T_M
                      else inv_sum_inv_sigma[tup] * (sum_inv_matvec[tup] - lambda_map[tup]) )
                      for tup in self.R_Q
                      if tup != ()
                     }
            alpha_star[()] = inv_sum_inv_sigma[()] * (sum_inv_matvec[()] - lambda_map[()])
            # calc mu
            mu = {
                tup : torch.stack([ ( (self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv() ) @ alpha_star[cols] )
                         for cols in downward_closure([tup])]).sum(dim = 0)
                for tup in self.T_Q
             }
            
            # calc objective
            objective_main = torch.stack([ 
                (1/self.covar[idx][0]) * ((self.y[idx] - alpha_star[tup]) @ (self.covar[idx][1].pinv() @ (self.y[idx] - alpha_star[tup])))
                if tup != ()
                else ((1/self.covar[idx][0]) * (self.y[idx] - alpha_star[tup]) ** 2)[0]
                for idx, tup in enumerate(self.T_M) 
            ]).sum().item()
            objective_lagr = torch.stack([
                lam_dict[tup] @ mu[tup]
                for tup in self.T_Q
            ]).sum().item()
            objective = objective_main + objective_lagr
            # print(objective, - objective_lagr)
            print(objective_main, - objective_lagr)
            obj_stats.append(objective)
            obj_main_stats.append(objective_main)
            dual_gap_stats.append(-objective_lagr)
            if math.isnan(objective):
                return('rerun')
            
            # update lambda
            lam_dict = {tup : torch.minimum(lam_dict[tup] + t * mu[tup], torch.zeros(lam_dict[tup].shape)) for tup in self.T_Q }
            
            if i % 100 == 0:
                marg_stats_i = [torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv() ) @ alpha_star[cols])
                         for cols in downward_closure([tup])]).sum(dim = 0).min().item() for tup in self.T_Q]
                print(marg_stats_i)
                marg_stats.append(marg_stats_i)
                marg_nnsum_i = [torch.minimum(torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv()) @ alpha_star[cols])
                         for cols in downward_closure([tup])]).sum(dim = 0), torch.zeros(getMarginalLength(tup, self.domain))).sum().item() for tup in self.T_Q]
                print(marg_nnsum_i)
                marg_nnsum.append(marg_nnsum_i)
                if (i > 200) and (np.sum(marg_nnsum_i) > -1):
                    break
                if true_answers:
                    full_y_opt = [alpha_star[tup] for tup in self.R_Q]
                    inferred_full = (self.marginals @ self.residuals_all.pinv()) @ full_y_opt
                    errors_full = np.mean([
                        torch.linalg.vector_norm((inferred_full[idx] - true_answers[idx]), 1).item() / num_records for idx in range(len(true_answers))
                    ])
                    error_stats.append(errors_full)
                    print('Current Error: ' + str(round(errors_full, 3)))
                    inferred_total = ((MarginalWorkload((), self.domain) @ self.residuals_all.pinv()) @ full_y_opt).item()
                    print(inferred_total)
                    total_stats.append(inferred_total)
            
            if early_stopping and (i > 400):
                if (abs(dual_gap_stats[-1]) < early_stopping):
                    break
                    
        marg_stats_i = [torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv() ) @ alpha_star[cols])
                         for cols in downward_closure([tup])]).sum(dim = 0).min().item() for tup in self.T_Q]
        print(marg_stats_i)
        marg_stats.append(marg_stats_i)
        marg_nnsum_i = [torch.minimum(torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv()) @ alpha_star[cols])
                 for cols in downward_closure([tup])]).sum(dim = 0), torch.zeros(getMarginalLength(tup, self.domain))).sum().item() for tup in self.T_Q]
        print(marg_nnsum_i)
        marg_nnsum.append(marg_nnsum_i)
        self.y_opt_dict = { tup : alpha_star[tup] for tup in self.R_Q }
        self.running_time = round(time.time() - start_time, 4)
        self.stats = [
                      marg_stats,
                      marg_nnsum,
                      obj_stats,
                      total_stats
                     ]
        if true_answers:
            self.stats.append(error_stats)
        return('done')
       
    def solveLooping(self, rounds = 4001, lam = -0.1, t = 1, t_div = 10, early_stopping = 0.01, true_answers = None, num_records = None, reg_param = 1):
        result = None
        while result != 'done':
            result = self.solve_regularization(rounds = rounds, 
                                lam = lam, 
                                t = t, 
                                early_stopping = early_stopping, 
                                true_answers = true_answers, 
                                num_records = num_records,
                                reg_param = reg_param)
            t = t / t_div
            
    def solve_regularization(self, rounds = 2001, lam = -0.1, t = 0.01, early_stopping = 0.01, true_answers = None, num_records = None, reg_param = 1):
        self.rounds = rounds
        self.t = t
        self.covar = [(self.sigmas[idx] ** 2, getCovarianceMat(tup, self.domain)) for idx, tup in enumerate(self.T_M)]
        marg_stats = []
        marg_nnsum = []
        obj_stats = []
        obj_main_stats = []
        error_stats = []
        dual_gap_stats = []
        total_stats = []
        start_time = time.time()
        
        lam_dict = { tup : lam * torch.ones((getMarginalLength(tup, self.domain),)) for tup in self.T_Q }
        
        
        inv_sum_inv_sigma = { tup : (1/(np.sum([(1 / self.covar[idx][0]) for idx, cols in enumerate(self.T_M) if cols == tup])) 
                                     if tup in self.T_M else 1)
                     for tup in self.R_Q 
                    }
        
        # sum_inv_matvec (handle total query separately)
        sum_inv_matvec = { tup : (torch.stack([ (1/self.covar[idx][0]) * (self.covar[idx][1].pinv() @ self.y[idx]) 
                                               for idx, cols in enumerate(self.T_M) if cols == tup]).sum(dim = 0) if tup in self.T_M
                          else torch.zeros((getResidualLength(tup, self.domain),)))
                          for tup in self.R_Q 
                          if tup != ()
                         }
        sum_inv_matvec[()] = torch.stack( [ self.y[idx] * (1/self.covar[idx][0]) for idx, cols in enumerate(self.T_M) if cols == ()]).sum(dim = 0)
        
        for i in range(rounds):
            # update lambda map
            lambda_map = { tup : torch.stack([ ( (self.marginals.workloads[self.T_Q.index(cols)] @ self.residuals_all.workloads[self.R_Q.index(tup)].pinv() ).T() @ lam_dict[cols] )
                                              for cols in self.T_Q if set(tup).issubset(set(cols))]).sum(dim = 0)
                          for tup in self.R_Q
                         }
            # calc new alpha star
            alpha_star = { 
                tup : 
                (inv_sum_inv_sigma[tup] * (self.covar[self.T_M.index(tup)][1] @ (sum_inv_matvec[tup] - lambda_map[tup]))
                      if tup in self.T_M
                      else  - 0.5 * reg_param * (((MarginalWorkload(tup, self.domain) @ ResidualWorkload2(tup, self.domain).pinv()).T() @ ((MarginalWorkload(tup, self.domain) @ ResidualWorkload2(tup, self.domain).pinv()))).pinv() @ lambda_map[tup])) 
                      for tup in self.R_Q
                      if tup != ()
                     }
            alpha_star[()] = max(inv_sum_inv_sigma[()] * (sum_inv_matvec[()] - lambda_map[()]), 0)
            # calc mu
            mu = {
                tup : torch.stack([ ( (self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv() ) @ alpha_star[cols] )
                         for cols in downward_closure([tup])]).sum(dim = 0)
                for tup in self.T_Q
             }
            
            # calc objective
            objective_main = torch.stack([ 
                (1/self.covar[idx][0]) * ((self.y[idx] - alpha_star[tup]) @ (self.covar[idx][1].pinv() @ (self.y[idx] - alpha_star[tup])))
                if tup != ()
                else ((1/self.covar[idx][0]) * (self.y[idx] - alpha_star[tup]) ** 2)[0]
                for idx, tup in enumerate(self.T_M) 
            ]).sum().item()
            objective_lagr = torch.stack([
                lam_dict[tup] @ mu[tup]
                for tup in self.T_Q
            ]).sum().item()
            objective = objective_main + objective_lagr
            print(objective_main, - objective_lagr)
            obj_stats.append(objective)
            obj_main_stats.append(objective_main)
            dual_gap_stats.append(-objective_lagr)
            if math.isnan(objective):
                return('rerun')
            
            # update lambda
            lam_dict = {tup : torch.minimum(lam_dict[tup] + t * mu[tup], torch.zeros(lam_dict[tup].shape)) for tup in self.T_Q }
            
            if i % 100 == 0:
                marg_stats_i = [torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv() ) @ alpha_star[cols])
                         for cols in downward_closure([tup])]).sum(dim = 0).min().item() for tup in self.T_Q]
                print(marg_stats_i)
                marg_stats.append(marg_stats_i)
                marg_nnsum_i = [torch.minimum(torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv()) @ alpha_star[cols])
                         for cols in downward_closure([tup])]).sum(dim = 0), torch.zeros(getMarginalLength(tup, self.domain))).sum().item() for tup in self.T_Q]
                print(marg_nnsum_i)
                marg_nnsum.append(marg_nnsum_i)
                if (i > 200) and (np.sum(marg_nnsum_i) > -1):
                    break
                if true_answers:
                    full_y_opt = [alpha_star[tup] for tup in self.R_Q]
                    inferred_full = (self.marginals @ self.residuals_all.pinv()) @ full_y_opt
                    errors_full = np.mean([
                        torch.linalg.vector_norm((inferred_full[idx] - true_answers[idx]), 1).item() / num_records for idx in range(len(true_answers))
                    ])
                    error_stats.append(errors_full)
                    print('Current Error: ' + str(round(errors_full, 3)))
                    inferred_total = ((MarginalWorkload((), self.domain) @ self.residuals_all.pinv()) @ full_y_opt).item()
                    print(inferred_total)
                    total_stats.append(inferred_total)
            
            if early_stopping and (i > 400):
                if (abs(dual_gap_stats[-1]) < early_stopping):
                    break
                     
        marg_stats_i = [torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv() ) @ alpha_star[cols])
                         for cols in downward_closure([tup])]).sum(dim = 0).min().item() for tup in self.T_Q]
        print(marg_stats_i)
        marg_stats.append(marg_stats_i)
        marg_nnsum_i = [torch.minimum(torch.stack([((self.marginals.workloads[self.T_Q.index(tup)] @ self.residuals_all.workloads[self.R_Q.index(cols)].pinv()) @ alpha_star[cols])
                 for cols in downward_closure([tup])]).sum(dim = 0), torch.zeros(getMarginalLength(tup, self.domain))).sum().item() for tup in self.T_Q]
        print(marg_nnsum_i)
        marg_nnsum.append(marg_nnsum_i)
        self.y_opt_dict = { tup : alpha_star[tup] for tup in self.R_Q }
        self.running_time = round(time.time() - start_time, 4)
        self.stats = [
                      marg_stats,
                      marg_nnsum,
                      obj_stats,
                      total_stats
                     ]
        if true_answers:
            self.stats.append(error_stats)
        return('done')
     