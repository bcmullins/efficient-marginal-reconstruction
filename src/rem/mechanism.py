from rem.reconstruction import gremMLE
from rem.utils import exponential, getOptimalSigmasCF, downward_closure
from rem.algebra import MarginalWorkload, ResidualWorkload2, VStack
from tqdm import tqdm
import numpy as np

class residualPlanner:
    def __init__(self, data, target_marginals, rho):
        """
        Instantiates Residual Planner mechanism with L2 Loss
        :param target_marginals: list of tuples of indices
        :param rho: scalar; privacy budget
        :param rounds: int
        """
        
        self.data = data
        self.target_marginal_tups = target_marginals
        self.rho = rho
        self.residual_tups, self.z_sigmas = getOptimalSigmasCF(marginals = self.target_marginal_tups,
                                         rho = self.rho, 
                                         domain = self.data.domain)
        self.residuals = VStack([ResidualWorkload2(tup, self.data.domain) for tup in self.residual_tups])
        self.z = self.residuals.getAnswers(self.data,  
                                           sigma = self.z_sigmas)
    def getMarginals(self):
        target_marginals = VStack([MarginalWorkload(tup, self.data.domain) for tup in self.target_marginal_tups])
        self.y = (target_marginals @ self.residuals.pinv()) @ self.z
        return self.y
        

class scalableMWEM:
    def __init__(self, target_marginals, rho, rounds):
        """
        Instantiates Scalable MWEM mechanism using GReM-MLE
        :param target_marginals: list of tuples of indices
        :param rho: scalar; privacy budget
        :param rounds: int
        """
        
        self.target_marginals = target_marginals
        self.rounds = rounds
        self.rho = rho
        self.gamma = 0.1
        self.alpha = 0.5
        self.rho_init = self.rho * self.gamma
        self.rho_round = (self.rho - self.rho_init) / self.rounds
        self.initialization = 0
                 
    def run(self, data):
        ## get all initialization
        init_idx = [tup for tup in downward_closure(self.target_marginals.cols()) 
                    if len(tup) == self.initialization]
        
        ## create M, y, sigmas
        for idx, tup in enumerate(init_idx):
            init_marginal = MarginalWorkload(tup, data.domain)
            y_init, y_init_sigma = init_marginal.getAnswers(data, 
                                                            rho = self.rho_init/len(init_idx), 
                                                            return_sigma = True)
            init_residuals, z_init, z_init_sigmas = init_marginal.decomposeIntoResiduals(y = y_init, 
                                                                                         sigma = y_init_sigma)
            if idx == 0:
                residuals, z, z_sigmas = init_residuals, z_init, z_init_sigmas
                marginals, y, y_sigmas = VStack([init_marginal]), [y_init], [y_init_sigma]
            else:
                residuals += init_residuals
                z += z_init
                z_sigmas += z_init_sigmas
                marginals = marginals.append(init_marginal)
                y.append(y_init)
                y_sigmas.append(y_init_sigma)     
        
        candidates = self.target_marginals.cols()
        
        for t in tqdm(range(self.rounds)):            
            # get scores
            candidate_marginals = VStack([MarginalWorkload(tup, data.domain) for tup in candidates])
            true_candidate_answers = candidate_marginals.getAnswers(data, 
                                                                    rho = self.rho_round, 
                                                                    return_sigma = False)
            gMLE = gremMLE(target_marginals = candidate_marginals, 
                           residuals = residuals, 
                           res_answers = z, 
                           res_sigmas = z_sigmas)
            inferred_answers = gMLE.getMarginals()
            scores = np.array([np.linalg.norm(inferred_answer - true_candidate_answers[i], ord = 1) 
                               for i, inferred_answer in enumerate(inferred_answers)])

            # run exp mechanism and measure selected workload
            c_star = exponential(candidates = candidates, 
                                 scores = scores, 
                                 sensitivity = 1, 
                                 epsilon = (self.alpha * self.rho_round * 8) ** 0.5)
            print(c_star, scores[candidates.index(c_star)], scores.max())
            candidates.remove(c_star)
            c_star_wkload = MarginalWorkload(c_star, data.domain)
            y_t, y_t_sigma = c_star_wkload.getAnswers(data, 
                                                      rho = (1 - self.alpha) * self.rho_round, 
                                                      return_sigma = True)
            residuals_t, z_t, z_t_sigmas = c_star_wkload.decomposeIntoResiduals(y = y_t, 
                                                                                sigma = y_t_sigma)
            
            # record residuals and marginals
            residuals += residuals_t
            z += z_t
            z_sigmas += z_t_sigmas
            marginals = marginals.append(c_star_wkload)
            y.append(y_t)
            y_sigmas.append(y_t_sigma)

        self.measured_residual_output = (residuals, z, z_sigmas)
        self.measured_marginal_output = (marginals, y, y_sigmas)
        
    def getMarginals(self):
        gmle = gremMLE(target_marginals = self.target_marginals, 
                       residuals = self.measured_residual_output[0], 
                       res_answers = self.measured_residual_output[1], 
                       res_sigmas = self.measured_residual_output[2])
        return(gmle.getMarginals())
       