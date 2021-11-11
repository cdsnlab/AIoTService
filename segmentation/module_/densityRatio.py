import numpy as np
from scipy.spatial import distance_matrix

class DensityRatio:

    """
        * test_data: x_{t-1} -> f_{t-1}(x)   (shape: (data_len, feature_len))
        * train_data: x_{t} -> f_{t}(x)      (shape: (data_len, feature_len))
        * test_data and train_data are two consecutive windows
    """

    def __init__(self, test_data, train_data, alpha=0.):

        assert test_data.shape == train_data.shape

        self.__test = test_data
        self.__train = train_data
        self.__kernel_centers = self.__test
        self.__kernel_num = test_data.shape[0]
        self.__n = train_data.shape[0]
        self.__median_distance = self.median_distance(np.concatenate((self.__test, self.__train), axis=0))

        sigma_list = self.__median_distance * np.array([0.6, 0.8, 1.0, 1.2, 1.4])
        lambda_list = 10. ** np.array([-3, -2, -1, 0, 1])

        self._DensityRatio(test_data = self.__test, 
                            train_data = self.__train, 
                            alpha = alpha, 
                            sigma_list = sigma_list, 
                            lambda_list = lambda_list)
    
    def __call__(self, data, theta):
        return self.calculate_density_ratio(data, theta)

    def calculate_density_ratio(self, data, theta): # Input: data = (n, d), theta = (b, 1)
        phi_data = self.gaussian_kernel_matrix(data, self.kernel_centers, self.__sigma).T # (b, n)
        density_ratio = np.asarray(phi_data.dot(np.matrix(theta).T)).ravel()                    # (n, b)@(b, 1) -> (n, 1)

        return density_ratio                              # (1, n)
    
    def median_distance(self, x):
        
        assert x.shape[0] == 2*self.__n

        dists = distance_matrix(x, x)     # (n, n)
        dists = np.tril(dists).reshape((1, -1))
        mdistance = np.sqrt(0.5) * np.median(dists[dists>0]) # rbf_dot has factor of two in kernel

        return mdistance

    def _DensityRatio(self, test_data, train_data, alpha, sigma_list, lambda_list):
        
        sigma, lambda_ = self._LCV(test_data, train_data, alpha, sigma_list, lambda_list)
        
        phi_test = self.gaussian_kernel_matrix(test_data, self.__kernel_centers, sigma).T # (b, n)
        phi_train = self.gaussian_kernel_matrix(train_data, self.__kernel_centers, sigma).T # (b, n)

        H = alpha * (phi_test.T.dot(phi_test) / self.__n) + \
            (1 - alpha) * (phi_train.T.dot(phi_train) / self.__n) # (b, b)
        # h = np.matrix(phi_test.mean(axis=1)) # (b, 1)
        h = phi_test.mean(axis = 0).T

        theta = np.linalg.solve(H + np.identity(self.__kernel_num) * lambda_, h).ravel() # (b, b)@(b,1)-> (b,1)
        theta[theta<0] = 0

        self.__alpha = alpha
        self.__theta = theta

        self.__sigma = sigma
        self.__lambda = lambda_

        self.__phi_test = phi_test
        self.__phi_train = phi_train

    def _LCV(self, test_data, train_data, alpha, sigma_list, lambda_list):
        """
            Likelihood Cross Validation
            Efficient Computation of LOOCV Score for uLSIF
        """

        score_cv, _sigma_cv, _lambda_cv = np.inf, 0, 0

        # one_nT = np.matrix(np.ones(self.__n))       # (1, n)
        # one_bT = np.matrix(np.ones(self.__kernel_num))    # (1, b)

        for sigma_candidate in sigma_list:

            phi_test = self.gaussian_kernel_matrix(test_data, self.__kernel_centers, sigma_candidate).T # (b, n)
            phi_train = self.gaussian_kernel_matrix(train_data, self.__kernel_centers, sigma_candidate).T # (b, n)

            H = alpha * (phi_test.T.dot(phi_test) / self.__n) + \
                (1 - alpha) * (phi_train.T.dot(phi_train) / self.__n) # (b, b)
            # h = np.matrix(phi_test.mean(axis = 1)) # (b, 1)
            h = phi_test.mean(axis=0).T

            for lambda_candidate in lambda_list:

                B = H + np.identity(self.__kernel_num) * lambda_candidate * (self.__n-1) / self.__n       # (b, b)
                BinvPtr = np.linalg.solve(B, phi_train)                                           # (b, b)@(b, n) -> (b, n) # invCK_de
                PtrBinvPtr = np.multiply(phi_train, BinvPtr)                                      # (b, n) element-wise multiplication
                # denominator = self.__n*one_nT - one_bT@PtrBinvPtr                                 # (1, b)@(b, n) -> (1, n) # tmp
                denominator = (self.__n * np.ones(self.__n) - np.ones(self.__n).dot(PtrBinvPtr)).A1
                
                diagonal_B0 = np.diag((h.T.dot(BinvPtr)).A1 / denominator)                            # (1, b)@(b, n) -> (1, n) -> (n, n)
                B0 = np.linalg.solve(B, h.dot(np.matrix(np.ones(self.__n)))) + \
                    BinvPtr.dot(diagonal_B0)                           # (b,b)@(b, 1)@(1, n) -> (b,n)

                diagonal_B1 = np.diag((np.ones(self.__n).dot(np.multiply(phi_test, BinvPtr))).A1 / denominator)    # (1, b)@(b, n)->(1,n) -> (n,n)
                B1 = np.linalg.solve(B, phi_test) + BinvPtr.dot(diagonal_B1)                               # (b,n)@(n,n)->(b,n) // (b,b)@(b,n) -> (b,n)

                B2 = (self.__n - 1)*(self.__n * B0 - B1) / (self.__n * (self.__n - 1))                            # (b, n)
                B2[B2 < 0] = 0

                w_train = np.multiply(phi_train, B2).sum(axis = 0).T                                     # (1, b)@(b,n)->(1,n)->(n,1) 
                w_test = np.multiply(phi_test, B2).sum(axis = 0).T                                       # (1,b)@(b,n) -> (1,n) -> (n,1)
                
                # score = w_train.T@w_train/(2*self.__n) - one_nT@w_test/self.__n                       # (1,n)@(n,1) -> (1,1)
                score = (w_train.T.dot(w_train).A1 / 2 - w_test.sum(axis = 0)) / self.__n
                if score < score_cv:
                    score_cv = score
                    _sigma_cv = sigma_candidate
                    _lambda_cv = lambda_candidate

        return _sigma_cv, _lambda_cv

    def gaussian_kernel_matrix(self, data, centers, sigma):
        answer = [
            [gaussian_kernel(datum, center, sigma) for datum in data] for center in centers
        ]
        return np.matrix(answer) # (b, n)
    
    @property
    def test_data(self):
        return self.__test
    
    @property
    def train_data(self):
        return self.__train

    @property
    def _median_distance(self):
        return self.__median_distance

    @property
    def alpha(self):
        return self.__alpha

    @property
    def sigma(self):
        return self.__sigma

    @property
    def lambda_(self):
        return self.__lambda

    @property
    def phi_test(self):
        return self.__phi_test
    
    @property
    def phi_train(self):
        return self.__phi_train

    @property
    def kernel_centers(self):
        return self.__kernel_centers

    @property
    def kernel_num(self):
        return self.__kernel_num

    @property
    def n_sample(self):
        return self.__n
    
    @property
    def theta(self):
        return self.__theta

    @property   # KLIEP
    def KLDiv(self):
        g_x = self.calculate_density_ratio(self.__test, self.__theta) # (1, n)

        divergence = np.log(g_x).sum(axis = 0) / self.__n

        return divergence

    @property   # uLSIF, RuLSIF
    def PEDiv(self):
        g_x = self.calculate_density_ratio(self.__test, self.__theta) # (1, n)
        g_y = self.calculate_density_ratio(self.__train, self.__theta) # (1, n)

        divergence = (- self.alpha * (g_x.T.dot(g_x))/2 \
            - (1 - self.alpha) * (g_y.T.dot(g_y))/2 + g_x.sum(axis = 0)) / self.__n - 0.5

        return divergence

    @property
    def SEPDistance(self):
        g_x = self.calculate_density_ratio(self.__test, self.__theta)

        distance = max(0, 0.5 - g_x.mean())

        return distance

def gaussian_kernel(datum, center, sigma):
    return np.exp(-0.5 * (np.linalg.norm(datum - center)**2) / (sigma**2))