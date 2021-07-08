import numpy as np
from scipy.spatial import distance_matrix

class DensityRatio:

    def __init__(self, test_data, train_data, option='rulsif', alpha=0., sigma_list=None, lambda_list=None, kernel_num=100):
        
        if test_data.shape[0]!=train_data.shape[0]:
            raise ValueError("Different dimension of sample")
        
        if test_data.shape[1]!=train_data.shape[1]:
            raise ValueError("Different number of samples")

        self.__test=test_data       # (n, d) 
        self.__train=train_data     # (n, d)
        self.__option=option
        
        if sigma_list is None:
            self.__median_distance=self.median_distance(np.concatenate((self.__test, self.__train)))
            sigma_list=self.__median_distance*np.array([0.6, 0.8, 1.0, 1.2, 1.4])
        
        if lambda_list is None:
            lambda_list=np.logspace(-3, 1, 5)
        
        self.__test_n=self.__test.shape[0]                                  # n
        self.__train_n=self.__train.shape[0]                                # n
        self.__kernel_num=min(kernel_num, self.__test_n, self.__train_n)    # b (=n)
        self.__kernel_centers=self.__test       
        self.__minimum=min(self.__test_n, self.__train_n)                   # n

        self._DensityRatio(test_data=self.__test, 
                            train_data=self.__train, 
                            option=self.__option,
                            alpha=alpha, 
                            sigma_list=sigma_list, 
                            lambda_list=lambda_list
                            )
    
    def __call__(self, data, theta):
        return self.calculate_density_ratio(data, theta)

    def calculate_density_ratio(self, data, theta):
        phi_data = self.gaussian_kernel_matrix(data=data, centers=self.kernel_centers, sigma=self.__sigma) # (b, n)
        density_ratio=theta.T@phi_data # (1,n)@(b,n)->(1,n)
        return density_ratio
    
    def median_distance(self, x):
        """
            INPUT: x (n-by-d)
                distance_matrix (n-by-n)
        """
        dists=distance_matrix(x, x)     # (n, n)
        dists=np.tril(dists).ravel()
        l=[item for item in dists if item>0.]
        # return np.sqrt(0.5)*np.median(np.array(l)).item() # WHY?
        return np.median(np.array(l)).item() # WHY?

    def _DensityRatio(self, test_data, train_data, option, alpha, sigma_list, lambda_list):
        if len(sigma_list)==1 and len(lambda_list)==1:
            sigma, lambda_ = sigma_list[0], lambda_list[0]
        else:
            sigma, lambda_,= self._CV(test_data, train_data, option, alpha, sigma_list, lambda_list)
        
        phi_train=self.gaussian_kernel_matrix(data=train_data, centers=self.__kernel_centers, sigma=sigma) # (b, n)
        phi_test=self.gaussian_kernel_matrix(data=test_data, centers=self.__kernel_centers, sigma=sigma) # (b, n)

        H=alpha*(phi_test@(phi_test.T)/self.__test_n)+(1-alpha)*(phi_train@(phi_train.T)/self.__train_n) # (b, b)
        h=np.matrix(phi_test.mean(axis=1)) # (b, 1)

        if option=='rulsif':
            theta=np.linalg.solve(H+np.identity(self.__kernel_num)*lambda_, h) # (b, b)@(b,1)->(b,1)
            theta[theta<0]=0
        else:
            theta=(np.identity(self.__kernel_num)/lambda_)@h # (b, b)@(b,1)->(b,1)

        self.__alpha=alpha
        self.__theta=theta

        self.__sigma=sigma
        self.__lambda=lambda_

        self.__phi_test=phi_test
        self.__phi_train=phi_train

    def _CV(self, test_data, train_data, option, alpha, sigma_list, lambda_list):
        score_cv, _sigma_cv, _lambda_cv=np.inf, 0, 0

        one_nT=np.matrix(np.ones(self.__minimum)) # (1, n)
        one_bT=np.matrix(np.ones(self.__kernel_num)) # (1, b)

        for _, sigma_candidate in enumerate(sigma_list):

            phi_train=self.gaussian_kernel_matrix(data=train_data, centers=self.__kernel_centers, sigma=sigma_candidate) # (b, n)
            phi_test=self.gaussian_kernel_matrix(data=test_data, centers=self.__kernel_centers, sigma=sigma_candidate) # (b, n)

            H=alpha*(phi_test@(phi_test.T)/self.__test_n)+(1-alpha)*(phi_train@(phi_train.T)/self.__train_n) # (b, b)
            h=np.matrix(phi_test.mean(axis=1)) # (b, 1)

            for _, lambda_candidate in enumerate(lambda_list):

                B=H+np.identity(self.__kernel_num)*lambda_candidate*(self.__train_n-1)/self.__train_n # (b, b)
                
                BinvPtr=np.linalg.solve(B, phi_train) # (b, b)@(b, n) -> (b, n)
   
                PtrBinvPtr=np.multiply(phi_train, BinvPtr) # (b, n) element-wise multiplication

                denominator=self.__train_n*one_nT - \
                    one_bT@PtrBinvPtr # (1, b)@(b, n) -> (1, n)

                diagonal_B0=np.diag((h.T@BinvPtr).A1/denominator.A1) # (1, b)@(b, n) -> (1, n) -> (n, n)

                B0=np.linalg.solve(B, h@one_nT) + \
                    BinvPtr@diagonal_B0 # (b,b)@(b, 1)@(1, n) -> (b,n)

                diagonal_B1=np.diag((one_bT@(np.multiply(phi_test, BinvPtr))).A1/denominator.A1) # (1, b)@(b, n)->(1,n) -> (n,n)

                B1=np.linalg.solve(B, phi_test) + \
                    BinvPtr@diagonal_B1 # (b,n)@(n,n)->(b,n) // (b,b)@(b,n) -> (b,n)

                B2=(self.__train_n-1)*(self.__test_n*B0-B1)/(self.__train_n*(self.__test_n-1)) # (b, n)
                B2[B2<0]=0
                
                # if option=='rulsif':
                w_train=(one_bT@(np.multiply(phi_train, B2))).T # (1, b)@(b,n)->(1,n)->(n,1) 
                w_test=(one_bT@(np.multiply(phi_test, B2))).T # (1,b)@(b,n) -> (1,n) -> (n,1)
                score_=np.square(w_train).mean()/2 - \
                    w_test.mean() # (1,n)@(n,1) -> (1,1)
                # else:
                # w_train_SEP=(one_bT@(np.multiply(phi_train, B2))).T # (1, b)@(b,n)->(1,n)->(n,1)
                # score_=abs(w_train_SEP.mean())

                if score_ < score_cv:
                    score_cv=score_
                    _sigma_cv=sigma_candidate
                    _lambda_cv=lambda_candidate
                # if sidx==0 and lambda_candidate==0.1:
                #     print(sigma_candidate, lambda_candidate)
                #     print('phi_test',phi_test)
                #     print('phi_train', phi_train)
                #     print('H',H)
                #     print('h',h)
                #     print('B',B)
                #     print('BP', BinvPtr)
                #     print('PBP', PtrBinvPtr)
                #     print('denom', denominator)
                #     print('B0',B0)
                #     print('B1',B1)
                #     print('B2',B2)
                #     print("w_train", w_train)
                #     print("w_test", w_test)
                
                

                
                # score_candidate_SEP=(1.5*h_.T@h_/lambda_candidate).item() # (1,n)@(n,1)->(1,1)

                # scores[sidx, lidx]=score_candidate_RuLSIF

                # if score_candidate_RuLSIF < score_RuLSIF:
                #     score_RuLSIF=score_candidate_RuLSIF
                #     _sigma_RuLSIF=sigma_candidate
                #     _lambda_RuLSIF=lambda_candidate
                
                

        # return _sigma_RuLSIF, _lambda_RuLSIF, _sigma_SEP, _lambda_SEP#, scores
        return _sigma_cv, _lambda_cv#, scores

    def gaussian_kernel_matrix(self, data, centers, sigma):
        answer=[[gaussian_kernel(datum=datum, center=center, sigma=sigma) for datum in data] for center in centers]
        return np.matrix(answer) # (b,n)

    # @property
    # def cv_scores(self):
    #     return self.__CV
    
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
        return self.__minimum
    
    @property
    def theta(self):
        return self.__theta

    @property
    def KLDiv(self):
        g_x = self.calculate_density_ratio(self.__test, self.__theta) # (1, n)

        divergence=np.log(g_x).mean()

        return divergence

    @property
    def PEDiv(self):
        g_x = self.calculate_density_ratio(self.__test, self.__theta) # (1, n)
        g_y = self.calculate_density_ratio(self.__train, self.__theta) # (1, n)

        divergence= -self.alpha*(np.square(g_x)).mean()/2 - \
                (1-self.alpha)*(np.square(g_y)).mean()/2 + \
                g_x.mean() - 0.5

        return divergence

    @property
    def SEP(self):
        g_x = self.calculate_density_ratio(self.__test, self.__theta) # (1, n)
        # g_y = self.calculate_density_ratio(self.__train, self.__theta) # (1, n)
        # g_x = np.multiply(self.__theta_SEP, (self.__phi_train).prod(axis=0))
        divergence = max(0, 0.5-g_x.mean())

        return divergence

def gaussian_kernel(datum, center, sigma):
    return np.exp(-0.5*(np.linalg.norm(datum-center)**2)/(sigma**2))