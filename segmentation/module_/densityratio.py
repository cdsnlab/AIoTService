import numpy as np
from scipy.spatial import distance_matrix
import random as rand
from functools import partial

class Densratio: 
    """Densratio
    The densratio class estimates the density ratio r(x) = p(x) / q(x) from two-samples x1 and x2 generated from two unknown distributions p(x), q(x), respectively, where x1 and x2 are d-dimensional real numbers.
    """
    def __init__(self, x, y, alpha=0., sigma=None, lamb=None, kernel_num=100):
        """[summary]

        Args:
            x (array-like of float): 
                Numerator samples array. x is generated from p(x).
            y (array-like of float): 
                Denumerator samples array. y is generated from q(x).
            alpha (float or array-like, optional): 
                The alpha is a parameter that can adjust the mixing ratio r(x) = p(x)/(alpha*p(x)+(1-alpha)q(x))
                , and is set in the range of 0-1. 
                Defaults to 0.
            sigma (float or array-like, optional): 
                Bandwidth of kernel. If a value is set for sigma, that value is used for kernel bandwidth
                , and if a numerical array is set for sigma, Densratio selects the optimum value by using CV.
            lamb (float or array-like, optional): 
                Regularization parameter. If a value is set for lamb, that value is used for hyperparameter
                , and if a numerical array is set for lamb, Densratio selects the optimum value by using CV.
            kernel_num (int, optional): The number of kernels in the linear model. Defaults to 100.

        Raises:
            ValueError: [description]
        """        

        self.__x = transform_data(x)
        self.__y = transform_data(y)

        if self.__x.shape[1] != self.__y.shape[1]:
            raise ValueError("x and y must be same dimentions.")

        if sigma is None:
            # sigma = np.logspace(-3,1,9)
            sigma = median_distance(np.concatenate((self.__x, self.__y)))*np.array([0.6, 0.8, 1, 1.2, 1.4])

        if lamb is None:
            # lamb = np.logspace(-3,1,9)
            lamb = np.logspace(-3,1,5)

        self.__x_num_row = self.__x.shape[0]
        self.__y_num_row = self.__y.shape[0]
        self.__kernel_num = np.min(np.array([kernel_num, self.__x_num_row])).item() #kernel number is the minimum number of x's lines and the number of kernel.
        # self.__centers = np.array(rand.sample(list(self.__x),k=self.__kernel_num)) #randomly choose candidates of rbf kernel centroid.
        self.__centers = self.__x
        self.__n_minimum = min(self.__x_num_row, self.__y_num_row)
        # self.__kernel  = jit(partial(gauss_kernel,centers=self.__centers))

        self._RuLSIF(x = self.__x,
                     y = self.__y,
                     alpha = alpha,
                     s_sigma = np.atleast_1d(sigma),
                     s_lambda = np.atleast_1d(lamb),
                    )

    def __call__(self,val):
        """__call__ method 
        call calculate_density_ratio.
        Args:
            val (`float` or `array_like of float`): 

        Returns:
            array_like of float. Density ratio at input val. r(val)
        """
        return self.calculate_density_ratio(val)

    def calculate_density_ratio(self, val):
        """calculate_density_ratio method

        Args:
            val (`float` or `array-like of float`) : [description]

        Returns:
            array-like of float : Density ratio at input val. r(val)
        """        

        val = transform_data(val)
        phi_x = self.__kernel(val, sigma=self.__sigma)
        density_ratio = np.dot(phi_x, self.__weights)
        return density_ratio

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

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
    def phi_x(self):
        return self.__phi_x

    @property
    def phi_y(self):
        return self.__phi_y

    @property
    def kernel_centers(self):
        return self.__centers

    @property
    def N_kernels(self):
        return self.__kernel_num

    @property
    def weights(self):
        return self.__weights

    @property
    def weights_(self):
        return self.__weights_

    @property
    def KLDiv(self):
        return np.log(np.dot(self.__phi_x,self.__weights)).mean()

    @property
    def PEDiv(self):
        g_y=np.dot(self.__phi_y, self.__weights)
        g_x=np.dot(self.__phi_x, self.__weights)
        # return (-0.5*g_y.T.dot(g_y)+g_x.sum(axis=0))/self.__x_num_row-0.5
        return (-self.__alpha*(g_x.T.dot(g_x))/2 - (1-self.__alpha)*(g_y.T.dot(g_y))/2 + g_x.sum(axis=0))/self.__n_minimum-0.5

    @property
    def SEPDiv(self):
        g_x=np.dot(self.__phi_x, self.__weights_)
        # g_y=np.dot(self.__phi_y, self.__weights_)
        return max(0, 1-g_x.sum()/self.__n_minimum)
        # return max(0, 0.5-g_x.sum()/self.__n_minimum)

    #main
    def _RuLSIF(self,x,y,alpha,s_sigma,s_lambda):
        if len(s_sigma)==1 and len(s_lambda)==1:
            sigma = s_sigma[0]
            lambda_ = s_lambda[0]
        else:
            optimized_params = self._optimize_sigma_lambda(x,y,alpha,s_sigma,s_lambda)#; print(optimized_params)
            sigma = optimized_params['sigma']
            lambda_ = optimized_params['lambda']

        phi_x=gauss_kernel_(centers=self.__centers, input=x, param=sigma)   # (b, n)
        phi_y=gauss_kernel_(centers=self.__centers, input=y, param=sigma)   # (b, n)

        H = (1.- alpha)*(np.dot(phi_y.T, phi_y) / self.__y_num_row)+\
             alpha*(np.dot(phi_x.T, phi_x)/self.__x_num_row)
        h = np.average(phi_x, axis=1).T

        weights = np.linalg.solve(H + lambda_*np.identity(self.__kernel_num), h).ravel()
        weights[weights<0]=0

        h_=np.prod(phi_y, axis=1)/self.__n_minimum
        # h_=np.average(phi_y, axis=1).T # (1, b)
        # weights_ = np.linalg.solve(-lambda_*np.identity(self.__kernel_num), h_).ravel()
        # weights_ = -weights_
        weights_ = (-h_/lambda_).ravel()

        # weights__ = np.linalg.solve(-lambda_*np.identity(self.__kernel_num), h).ravel()
        # weights__ = -weights__

        self.__alpha = alpha
        self.__weights = weights
        self.__weights_ = -weights_
        # self.__weights__ = weights__
        self.__lambda = lambda_
        self.__sigma = sigma
        self.__phi_x = phi_x
        self.__phi_y = phi_y

    def _optimize_sigma_lambda(self,x,y,alpha,s_sigma,s_lambda):
        score_new = np.inf
        sigma_new = 0 
        lamb_new = 0
        
        b_, n_, ntr, nte=self.__kernel_num, self.__n_minimum, self.__y_num_row, self.__x_num_row
        ones_nt=np.ones(n_).reshape((1, n_))    # (1, n)
        ones_bt=np.ones(b_).reshape((1, b_))    # (1, b)
        for i, sig in enumerate(s_sigma):
            phi_x=gauss_kernel_(centers=self.__centers, input=x, param=sig)\
                .reshape((b_, n_)) #   (b, n)
            phi_y=gauss_kernel_(centers=self.__centers, input=y, param=sig)\
                .reshape((b_, n_)) #   (b, n)

            # H = (1.- alpha)*(np.dot(phi_y.T, phi_y) / self.__y_num_row) \
            #     + alpha*(np.dot(phi_x.T, phi_x)/self.__x_num_row)
            H = ((1.- alpha)*(np.dot(phi_y, phi_y.T) / ntr)\
                + alpha*(np.dot(phi_x, phi_x.T) / nte))\
                .reshape((b_, b_))    #   (b, b)
            # h = phi_x.mean(axis=0, keepdims=True).T
            h = (np.sum(phi_x, axis=1) / nte)\
                .reshape((b_, 1))                   #   (b, 1)

            # phi_x = phi_x[:self.__n_minimum].T
            # phi_y = phi_y[:self.__n_minimum].T

            for lam in s_lambda:
                B = ( H+np.identity(b_)*(lam*(ntr - 1)/ntr) ).reshape((b_, b_)) #   (b, b)
                
                BinvX = np.linalg.solve(B, phi_y).reshape((b_, n_))          #   (b, n)
                XBinvX = (phi_y * BinvX).reshape((b_, n_))                  #   (b, n)
                
                D0_d = ntr*ones_nt-np.dot(ones_bt, XBinvX)    #   (1, n)
                D0_n = np.dot(h.T, BinvX).reshape((1, n_))
                diag_D0 = np.diag((D0_n/D0_d).ravel())  # (n, n)
                B0 = np.linalg.solve(B, h*ones_nt) + np.dot(BinvX, diag_D0) # (b, n)

                D1_d, D1_n = D0_d, np.dot(ones_bt, phi_x*BinvX) # (1, n)
                diag_D1 = np.diag((D1_n/D1_d).ravel())  # (n, n)
                B1 = np.linalg.solve(B, phi_x) + np.dot(BinvX, diag_D1) # (b, n)

                B2 = (ntr - 1) * (nte * B0 - B1)/(ntr *(nte - 1))
                B2[B2<0]=0

                r_x = np.dot(ones_bt, phi_x*B2).T   # (n, 1)
                r_y = np.dot(ones_bt, phi_y*B2).T   # (n, 1)

                # r_x = (phi_x * B2).sum(axis=1).T
                # r_y = (phi_y * B2).sum(axis=1).T

                score = (np.dot(r_y.T,r_y).ravel()/(2.*n_) - np.dot(ones_nt, r_x)/n_).item() #LOOCV

                if (score < score_new):
                    score_new = score
                    sigma_new = sig
                    lamb_new = lam

        return {'sigma':sigma_new, 'lambda':lamb_new}

def median_distance(x):
    dists=distance_matrix(x, x)
    dists=np.tril(dists).ravel()
    l=[item for item in dists if item>0.]
    return np.median(np.array(l)).item()

# def gauss_kernel(r,centers,sigma):
#     dists = pairwise_distances(euclid_distance)
#     return np.exp(-0.5*dists(r,centers) / (sigma**2))
#     # return np.exp(-0.5*pairwise_euclid_distances(r,centers) / (sigma**2))

def gauss_kernel_(centers, input, param):
    """
        (b, n)
        if #centers = #input = n,
            row corresponds to a kernel center
            col corresponds to a input datum
    """
    return np.exp(-0.5*np.square(distance_matrix(centers, input)) / (param**2))

def transform_data(x):
    if isinstance(x,np.ndarray):
        if len(x.shape)==1:
            return np.atleast_2d(x.astype(np.float64)).T
        else:
            return np.atleast_2d(x.astype(np.float64))
    elif isinstance(x,list):
        return transform_data(np.array(x))
    else:
        raise ValueError("Cannot convert to numpy.array")

# def euclid_distance(x,y, square=True):
#     '''
#     \sum_m (X_m - Y_m)^2
#     '''
#     XX=np.dot(x,x)
#     YY=np.dot(y,y)
#     XY=np.dot(x,y)
#     if not square:
#         return np.sqrt(XX+YY-2*XY)
#     return XX+YY-2*XY

# def pairwise_distances(dist,**arg):
#     '''
#     d_ij = dist(X_i , Y_j)
#     "i,j" are assumed to indicate the data index.
#     '''
#     return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))

# def pairwise_euclid_distances(x,y,square=True):
#     XX = np.einsum('ik,ik->',x,x)
#     YY = np.einsum('ik,ik->',y,y)
#     XY = np.einsum('ik,jk->ij',x,y)
#     if not square:
#         return np.sqrt(XX[:,np.newaxis]+YY[np.newaxis,:] - 2*XY)
#     return XX[:,np.newaxis]+YY[np.newaxis,:] - 2*XY