import numpy as np
import sys
from scipy.spatial import distance_matrix

class DensityRatioSEP:

    """
        * ref_data: x_{t-1} -> f_{t-1}(x)   (shape: (data_len, feature_len))
        * test_data: x_{t} -> f_{t}(x)      (shape: (data_len, feature_len))
        * ref_data and test_data are two consecutive windows
    """

    def __init__(self, ref_data, test_data):

        assert ref_data.shape==test_data.shape

        self.__ref_data = ref_data
        self.__test_data = test_data
        self.__kernel_centers = ref_data
        self.__length = min(len(ref_data), len(test_data))

        # implementation configurations: https://www.sciencedirect.com/science/article/pii/S0893608013000270?via%3Dihub
        # Liu et al., Change-point detection in time-series data by relative density-ratio estimation
        median_distance = self.compute_median_distance(np.concatenate([ref_data, test_data], axis=0))

        sigma_range = median_distance * np.array([0.6, 0.8, 1., 1.2, 1.4])
        lambda_range = 10. ** np.array([-3, -2, -1, 0, 1])
        
        # the best combination is chosen by grid search via cross-validation
        sigma_, lambda_ = self.cross_validation(ref_data, test_data, sigma_range, lambda_range)
        self.__sigma_ = sigma_
        self.__lambda_ = lambda_

        self._SEP(ref_data, test_data)

    def compute_median_distance(self, samples):

        assert samples.shape[0] == 2*self.__length
        distances = distance_matrix(samples, samples)
        distances = np.tril(distances).reshape((1, -1))

        median_distance = np.sqrt(0.5) * np.median(distances[distances>0])

        return median_distance if median_distance != 0 and (not np.isnan(median_distance)) else 0.1
            
    def cross_validation(self, ref_data, test_data, sigmas, lambdas):

        n = self.__length
        optimal_sigma = optimal_lambda = 0.
        opt_score = sys.maxsize

        one_nT = np.matrix(np.ones(n))    # (1, n)
        one_bT = np.matrix(np.ones(n))    # (1, b)

        for sigma_ in sigmas:

            phi_ref = self.kernel_matrix(ref_data, self.__kernel_centers,  sigma_)
            phi_test = self.kernel_matrix(test_data, self.__kernel_centers,  sigma_)

            # H = phi_test@phi_test.T/n # (b, b)
            h_hat = np.mean(phi_test, axis=0).T

            for lambda_ in lambdas:

                # B = H + np.identity(n)*lambda_*(n-1)/n # (b, b)
                B = np.identity(n)*lambda_*(n-1)/n
            
                BinvPtr = np.linalg.solve(B, phi_test) # (b, b)@(b, n) -> (b, n) # invCK_de
                PtrBinvPtr = np.multiply(phi_test, BinvPtr) # (b, n) element-wise multiplication
                denominator = n*one_nT - one_bT@PtrBinvPtr # (1, b)@(b, n) -> (1, n) # tmp

                diagonal_B0 = np.diag((h_hat.T@BinvPtr).A1/denominator.A1) # (1, b)@(b, n) -> (1, n) -> (n, n)
                B0 = np.linalg.solve(B, h_hat@one_nT) + BinvPtr@diagonal_B0 # (b,b)@(b, 1)@(1, n) -> (b,n)

                diagonal_B1 = np.diag((one_bT@(np.multiply(phi_ref, BinvPtr))).A1/denominator.A1) # (1, b)@(b, n)->(1,n) -> (n,n)
                B1 = np.linalg.solve(B, phi_ref) + BinvPtr@diagonal_B1 # (b,n)@(n,n)->(b,n) // (b,b)@(b,n) -> (b,n)

                B2 = (n-1)*(n*B0-B1)/(n*(n-1)) # (b, n)
                B2[B2<0] = 0

                w_test = (one_bT@(np.multiply(phi_test, B2))).T # (1, b)@(b,n)->(1,n)->(n,1)
                w_ref = (one_bT@(np.multiply(phi_ref, B2))).T # (1,b)@(b,n) -> (1,n) -> (n,1)

                objective_score = abs(0.5 - w_ref.mean())

                if objective_score < opt_score:
                    opt_score = objective_score
                    optimal_sigma = sigma_
                    optimal_lambda = lambda_

        return optimal_sigma, optimal_lambda

    def _SEP(self, ref_data, test_data):
        
        n = self.__length

        phi_ref = self.kernel_matrix(ref_data, self.__kernel_centers, self.__sigma_)
        phi_test = self.kernel_matrix(test_data, self.__kernel_centers, self.__sigma_)

        assert phi_ref.shape==(self.__length, self.__length)

        h_hat = np.mean(phi_test, axis=0).T # mean of all centers of each test datum

        # theta = np.linalg.solve(np.identity(n)*self.__lambda_, h_hat)
        theta = h_hat/self.__lambda_
        theta[theta<0] = 0

        self.__kernel_matrix = phi_test
        self.__h_hat = h_hat
        self.__theta = theta

        density_ratio = phi_ref.T@theta

        assert density_ratio.shape==theta.shape

        score = max(0, 0.5 - density_ratio.mean())
        
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
