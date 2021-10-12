import numpy as np
import sys
from scipy.spatial import distance_matrix

class DensityRatioSEP: 

    # ref_data: x_{t-1} -> f_{t-1}(x)   (shape: (data_len, feature_len))
    # test_data: x_{t} -> f_{t}(x)      (shape: (data_len, feature_len))
    # ref_data and test_data are two consecutive windows

    def __init__(self, ref_data, test_data):

        assert ref_data.shape==test_data.shape

        self.__ref_data = ref_data
        self.__test_data = test_data
        self.__kernel_centers = ref_data
        self.__length = min(len(ref_data), len(test_data))

        total_samples = np.concatenate([ref_data, test_data], axis=0)
        median_distance = self.compute_median_distance(total_samples)
        sigma_range = median_distance * np.array([0.6, 0.8, 1., 1.2, 1.4])
        lambda_range = 10. ** np.array([-3, -2, -1, 0, 1])

        sigma_, lambda_ = self.cross_validation(ref_data, test_data, sigma_range, lambda_range)
        self.__sigma_ = sigma_
        self.__lambda_ = lambda_

        self._SEP(ref_data, test_data)

    def compute_median_distance(self, samples):
        
        distances = distance_matrix(samples, samples)
        distances = np.tril(distances).reshape((1, -1))

        median_distance = np.sqrt(0.5) * np.median(distances[distances>0])

        if np.isnan(median_distance):
            median_distance = 0.

        return median_distance if median_distance != 0. else 0.1
            
    def cross_validation(self, ref_data, test_data, sigmas, lambdas):
        # to determine theta (sigma and lambda) - leave-one-out
        # lambda is empirically chosen by cross_validation

        n = self.__length

        optimal_sigma = optimal_lambda = 0.
        opt_score = sys.maxsize

        one_nT=np.matrix(np.ones(n))    # (1, n)
        one_bT=np.matrix(np.ones(n))    # (1, b)

        # TODO: What is optimization term? - SEP score
        # find theta minimizing J and (sigma, lambda) of the theta 

        for sigma_ in sigmas:

            phi_ref = self.kernel_matrix(ref_data, self.__kernel_centers,  sigma_)
            phi_test = self.kernel_matrix(test_data, self.__kernel_centers,  sigma_)
            assert phi_ref.shape==(self.__length, self.__length)

            # h_hat = np.prod(phi_test, axis=0).reshape((-1, 1))#/self.__length # (1, n)
            # h_hat = (np.prod(phi_test, axis=1)/self.__length).reshape((1, -1)) # 1-by-test
            # h_hat = np.mean(phi_test, axis=1).reshape((-1, 1)) # (1, n), https://link.springer.com/content/pdf/10.1007/s40860-018-0065-2.pdf
            # assert h_hat.shape==(1, self.__length)

            h_hat = np.mean(phi_test, axis=0).T

            for lambda_ in lambdas:

                B=-np.identity(n)*lambda_*(n-1)/n # (b, b)
                # B=(-lambda_candidate*np.identity(self.__kernel_num))*(self.__train_n-1)/self.__train_n
                BinvPtr=np.linalg.solve(B, phi_test) # (b, b)@(b, n) -> (b, n) # invCK_de
                # beta = np.linalg.solve(B, h) # (b, b)@(b, 1) -> (b, 1)
                PtrBinvPtr=np.multiply(phi_test, BinvPtr) # (b, n) element-wise multiplication
                denominator=n*one_nT - one_bT@PtrBinvPtr # (1, b)@(b, n) -> (1, n) # tmp

                diagonal_B0=np.diag((h_hat.T@BinvPtr).A1/denominator.A1) # (1, b)@(b, n) -> (1, n) -> (n, n)
                # diagonal_B0=np.diag((beta.T@phi_train).A1/denominator.A1) # (1, b)@(b, n) -> (1, n) -> (n, n)
                B0=np.linalg.solve(B, h_hat@one_nT) + BinvPtr@diagonal_B0 # (b,b)@(b, 1)@(1, n) -> (b,n)

                diagonal_B1=np.diag((one_bT@(np.multiply(phi_ref, BinvPtr))).A1/denominator.A1) # (1, b)@(b, n)->(1,n) -> (n,n)
                B1=np.linalg.solve(B, phi_ref) + BinvPtr@diagonal_B1 # (b,n)@(n,n)->(b,n) // (b,b)@(b,n) -> (b,n)

                B2=(n-1)*(n*B0-B1)/(n*(n-1)) # (b, n)
                B2[B2<0]=0

                w_test=(one_bT@(np.multiply(phi_test, B2))).T # (1, b)@(b,n)->(1,n)->(n,1) 
                w_ref=(one_bT@(np.multiply(phi_ref, B2))).T # (1,b)@(b,n) -> (1,n) -> (n,1)

                objective_score = max(0, 0.5-w_ref.mean())

                # theta = (h_hat/lambda_) *-1 # (1, n)
                # right_term = np.prod(phi_ref, axis=0).reshape((1, -1))
                # assert theta.shape==right_term.shape
                # density_ratio = np.multiply(theta, right_term).reshape(1,-1)
                # # right_term = np.prod(phi_test, axis=1)
                # # right_term = np.diagflat(right_term)
                # # assert right_term.shape==(n, n)
                # # density_ratio = theta@phi_test.T # (1, n)@(n, n) => (1, n)
                # # density_ratio = kernel
                # # assert density_ratio.shape==(1, n)
                # objective_score = max(0, 0.5 - density_ratio.mean())
                # # objective_score = abs(0.5-np.sum(density_ratio)/n)

                if objective_score < opt_score:
                    opt_score = objective_score
                    optimal_sigma = sigma_
                    optimal_lambda = lambda_

        return optimal_sigma, optimal_lambda

    def _SEP(self, ref_data, test_data):
        
        n = self.__length

        # phi_ref = self.kernel_matrix(self.__kernel_centers, ref_data, self.__sigma_)
        phi_ref = self.kernel_matrix(ref_data, self.__kernel_centers, self.__sigma_)
        # phi_test = self.kernel_matrix(self.__kernel_centers, test_data, self.__sigma_)
        phi_test = self.kernel_matrix(test_data, self.__kernel_centers, self.__sigma_)
        assert phi_ref.shape==(self.__length, self.__length)

        # h_hat = (np.prod(phi_test, axis=1)/self.__length).reshape((1, -1)) # 1-by-test
        # h_hat = np.mean(phi_ref, axis=1).reshape((1,-1))
        # h_hat = np.mean(phi_test, axis=1).reshape((-1, 1)) # (n, 1), https://link.springer.com/content/pdf/10.1007/s40860-018-0065-2.pdf
        # h_hat = np.prod(phi_test, axis=0).reshape((-1, 1))#/self.__length # (1, n)

        h_hat = np.mean(phi_test, axis=0).T # mean of all centers of each test datum

        # theta = h_hat/self.__lambda_ #* -1
        theta = np.linalg.solve(np.identity(n)*self.__lambda_, h_hat)
        # theta[theta<0] = 0

        self.__kernel_matrix = phi_test
        self.__h_hat = h_hat
        self.__theta = theta

        # density_ratio = phi_test.T@theta

        density_ratio = np.multiply(
            np.prod(phi_ref, axis=0), theta
        )

        assert density_ratio.shape==theta.shape

        # right_term = np.prod(phi_test, axis=0).reshape((1, -1))
        # right_term = np.diagflat(right_term)
        # assert right_term.shape==(n, n)    
        # density_ratio = np.multiply(theta, right_term).reshape(1,-1)
        # density_ratio = theta@phi_test.T # (1, n)@(n, n) => (1, n)

        score = max(0, 0.5 - density_ratio.mean())
        # score = abs(0.5 - np.sum(density_ratio) / n)
        
        self.__density_ratio = density_ratio
        self.__score = score
        

    def kernel_matrix(self, data, centers, sigma):

        answer = [
            [rbf_kernel_function(d, c, sigma) for d in data] for c in centers
        ]

        return np.matrix(answer) # C x D

    @property
    def _kernel_matrix(self):
        return self.__kernel_matrix

    @property
    def h_hat(self):
        return self.__h_hat

    @property
    def kernel_centers(self):
        return self.__kernel_centers

    @property
    def theta(self):
        return self.__theta

    @property
    def score(self):
        return self.__score

    @property
    def density_ratio(self):
        return self.__density_ratio

    @property
    def sigma_(self):
        return self.__sigma_

    @property
    def lambda_(self):
        return self.__lambda_

    @property
    def ref_data(self):
        return self.__ref_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def length(self):
        return self.__length

def rbf_kernel_function(data, center, sigma):

    r = 1./(2.*(sigma**2))
    answer = np.exp(
        -r * (sum((data-center) ** 2))
    )

    return answer
