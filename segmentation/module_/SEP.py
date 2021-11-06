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
        sigma_, lambda_ = self._CV(ref_data, test_data, sigma_range, lambda_range)
        self.__sigma_ = sigma_
        self.__lambda_ = lambda_

        self._SEP(ref_data, test_data)

    def compute_median_distance(self, samples):

        assert samples.shape[0] == 2*self.__length
        distances = distance_matrix(samples, samples)
        distances = np.tril(distances).reshape((1, -1))

        median_distance = np.sqrt(0.5) * np.median(distances[distances>0])

        return median_distance

    def _CV(self, ref_data, test_data, sigmas, lambdas):

        n = self.__length

        fold = 2

        score_cv = np.zeros((len(sigmas), len(lambdas)))
        cv_index_nu = np.random.mtrand.permutation(n)
        cv_split_nu = np.floor(np.r_[0:n] * fold / n)

        for sigma_index in np.r_[0:len(sigmas)]:
            sigma = sigmas[sigma_index]
            phi_test = self.kernel_matrix(test_data, ref_data, sigma)
            score_tmp = np.zeros((fold, len(lambdas)))

            for k in np.r_[0:fold]:
                mKtmp = np.mean(phi_test[:, cv_index_nu[cv_split_nu != k]], 1) # k-index에 위치한 값을 제외하고 전부 (h)

                for lambda_index in np.r_[0:len(lambdas)]:
                    lbd = lambdas[lambda_index]
                    thetah_cv = np.linalg.solve(lbd * np.eye(n), mKtmp)

                    cv_g_x = np.mean(np.dot(mKtmp.T, thetah_cv).T)
                    score_tmp[k, lambda_index] = abs(1 - cv_g_x)
                
                score_cv[sigma_index, :] = np.mean(score_tmp, 0)

        score_cv_tmp = score_cv.min(1)
        lambda_chosen_index, sigma_chosen_index = score_cv.argmin(1), score_cv_tmp.argmin()
        lambda_chosen = lambdas[lambda_chosen_index[sigma_chosen_index]]
        sigma_chosen = sigmas[sigma_chosen_index]

        return sigma_chosen, lambda_chosen
            
    def cross_validation(self, ref_data, test_data, sigmas, lambdas):

        n = self.__length
        optimal_sigma = optimal_lambda = 0.
        opt_score = sys.maxsize

        for sigma_ in sigmas:

            for lambda_ in lambdas:

                objective_score = 0.
                for i in range(n):
                    ref_i = np.concatenate((ref_data[:i], ref_data[i+1:]), axis=0)
                    test_i = np.concatenate((test_data[:i], test_data[i+1:]), axis=0)

                    phi_ref_i = self.kernel_matrix(ref_i, ref_i, sigma_)
                    phi_test_i = self.kernel_matrix(test_i, ref_i, sigma_)

                    h_hat_i = np.mean(phi_test_i, axis=0).T
                    # h_hat_i = np.mean(phi_test_i, axis=1)
                    theta = h_hat_i/lambda_

                    w_test = abs(phi_test_i*theta)
                    
                    # J = abs(h_hat_i*theta) + lambda_*(theta**2)/2
                    objective_score += w_test
                
                objective_score /= n

                if objective_score < opt_score:
                    opt_score = objective_score
                    optimal_sigma = sigma_
                    optimal_lambda = lambda_

        return optimal_sigma, optimal_lambda

    def _SEP(self, ref_data, test_data):

        phi_ref = self.kernel_matrix(ref_data, ref_data, self.__sigma_)
        phi_test = self.kernel_matrix(test_data, ref_data, self.__sigma_)

        coe = self.__lambda_ * np.eye(self.__length)
        var = np.mean(phi_test, 1)

        thetat = np.linalg.solve(coe, var)
        # wh_x_nu = np.dot(phi_test.T, thetat).T
        wh_x_nu = np.dot(phi_ref.T, thetat).T

        score = max(0, 1 - np.mean(wh_x_nu))

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
