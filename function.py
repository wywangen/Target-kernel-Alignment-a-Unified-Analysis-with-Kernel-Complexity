import random
import numpy as np
from scipy import stats as st
from sklearn.metrics import mean_squared_error
from cvxopt import matrix, solvers


def Kernel_rbf(x_1,x_2,sigma=1.0):
    """function to calculate Gaussian kernel

    parameters
    ----------
    x1: array-like of shape (n_samples_1, n_features)

    x2: array-like of shape (n_samples_2, n_features)

    sigma: float, default=1.0

    Returns
    -------
    kernel: array-like of shape (n_samples_1, n_samples_2)
            kernel matrix of x1 and x2
    """
    if x_1.ndim==1:
        x_1=x_1.reshape(x_1.shape[0],1)
    if x_2.ndim==1:
        x_2=x_2.reshape(x_2.shape[0],1)
        
    dist_sq=np.sum(x_1**2,1).reshape(-1,1)+np.sum(x_2**2,1)-2*np.dot(x_1,x_2.T)
    K=np.exp(-sigma * dist_sq)
    return K


def Kernel_sobo(x_1,x_2):
    """
    function to calculate first sobole space kernel of scalar input
    
    parameters
    ----------
    x1: array-like of shape (n_samples_1, )

    x2: array-like of shape (n_samples_2, )

    Returns
    -------
    kernel: array-like of shape (n_samples_1, n_samples_2)
            kernel matrix of x1 and x2
    """

    # Create meshgrid of indices
    i, j = np.meshgrid(np.arange(len(x_1)), np.arange(len(x_2)))

    # Get matrix with largest elements
    Kernel_matrix = np.minimum(x_1[i], x_2[j])

    return Kernel_matrix.T


def Kernel_laplace(x_1, x_2):
    # Create meshgrid of indices
    i, j = np.meshgrid(np.arange(len(x_1)), np.arange(len(x_2)))

    # Get matrix with largest elements
    Kernel_matrix = 0.5*np.exp(-abs(x_1[i] - x_2[j]))
    
    return Kernel_matrix.T

def laplace_kernel(X, Y):
    n, p = X.shape
    m, _ = Y.shape
    output = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            output[i, j] = np.exp(-np.sum(np.abs(X[i] - Y[j])))
    
    return output

def Kernel_poly(x_1, x_2, gamma):
    """
    this is kernel for multiple dimension data
    """

    return (1+gamma*np.dot(x_1,x_2.T))**5



def kernel_truncation(K, r, pre_SVD=None):
    """function to truncate kernel matrix

    parameters
    ----------
    x: array-like of shape (n_test_samples, n_features)
        new data point

    K: array-like of shape (n_train_samples, n_samples)
        kernel matrix
    

    r: int
        truncation parameter

    Returns
    -------
    K_truncated_matrix: array-like of shape (n_samples, n_samples)
                    truncated kernel matrix
    """
    if pre_SVD==None:
        U, s, V = np.linalg.svd(K)
    else:
        U, s, V = pre_SVD
    K_truncated_matrix = U[:, :r].dot(np.diag(s[:r])).dot(V[:r, :])

    return K_truncated_matrix




def KRR_estimation(K, y_train, lam, truncation=False, r=None, pre_SVD=None):
    """function to accomplish estimation for kernel ridge regression
    
    parameters
    ----------
    x: array-like of shape (n_samples, n_features)
    x_train: array-like of shape (n_samples_1, n_features)
    y_train: array-like of shape (n_samples_1, )
    
    return
    ------
    result: array-like of shape (n_samples, )
            estimation value of estimated function on the training set
    """
    n=K.shape[0]
    def full_KRR(K, y, lam):
        """use full kernel matrix to accomplish estimation for kernel ridge regression
        
        parameters
        ----------
        K: array-like of shape (n_samples, n_samples)
            kernel matrix of x_train
        K_x: array-like of shape (n_samples, n_samples_1)
            kernel matrix between x and x_train
        """
        result = K.dot(np.linalg.inv(K+n*lam*np.identity(n)).dot(y))
        return result
    if truncation==False:
        result = full_KRR(K, y_train, lam)
    else:
        if pre_SVD==None:
            K_approx = kernel_truncation(K, r)
        else:
            K_approx = kernel_truncation(K, r, pre_SVD)
        result = full_KRR(K_approx, y_train, lam)

    return result


def choose_lam_r(K, y_train, y_true, truncation=False):
    """
    function that return the best mse under specified lambda and r
    
    parameters
    ----------
    K: array-like of shape (n_samples, n_samples)
        kernel matrix of x_train
    
    y_train: array-like of shape (n_samples, )

    lam: array-like of shape (lam_list, )
        regularization parameter, if none then choose from defalt set

    r: array-like of shape (r_list, )
        truncation parameter, if none then choose from defalt set

    truncation: bool

    return
    ------
    optimal mse
    """
    n = K.shape[0]
    lam_list = np.linspace(0.0001,1,20)
    if truncation:
        r_list = 10**np.linspace(-3, 0, 20)
        error_list = np.zeros([len(lam_list), len(r_list)])
        U, s, V = np.linalg.svd(K)
        for i in range(len(lam_list)):
            for j in range(len(r_list)):
                result = KRR_estimation(K, y_train, lam_list[i], truncation=truncation,
                                         r=int(r_list[j]*n), pre_SVD=(U, s, V))
                error_list[i, j] = mean_squared_error(result, y_true)
        optimal_error = np.min(error_list)
    else:
        error_list = np.zeros(len(lam_list))
        for i in range(len(lam_list)):
            result = KRR_estimation(K, y_train, lam_list[i], truncation=truncation, r=None)
            error_list[i] = mean_squared_error(result, y_true)
        optimal_error = np.min(error_list)
    return optimal_error


def choose_lam(K, y_train, y_true, r, truncation=False):
    """
    function that return the best mse under specified lambda and r
    
    parameters
    ----------
    K: array-like of shape (n_samples, n_samples)
        kernel matrix of x_train
    
    y_train: array-like of shape (n_samples, )

    lam: array-like of shape (lam_list, )
        regularization parameter, if none then choose from defalt set

    r: array-like of shape (r_list, )
        truncation parameter, if none then choose from defalt set

    truncation: bool

    return
    ------
    optimal mse
    """
    lam = np.linspace(0.0001,1,20)
    mse = np.zeros(len(lam))
    U, s, V = np.linalg.svd(K)
    for i in range(len(lam)):
        y_est = KRR_estimation(K, y_train, lam[i], truncation=truncation, r=r, pre_SVD=(U, s, V))
        mse[i] = mean_squared_error(y_true, y_est)
    optimal_mse = np.min(mse)
    return optimal_mse




def l_tau(x,tau):
    """function to calculate check loss for KQR"""
    re=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i]>=0:
            re[i]=tau*x[i]
        else:
            re[i]=(1-tau)*x[i]
    return re


def empirical_KQR(y,f_hat,f_true,tau):
    """function to calculate empirical excess risk for KQR"""
    y=y.reshape(y.shape[0],)
    f_hat=f_hat.reshape(f_hat.shape[0],)
    f_true=f_true.reshape(f_true.shape[0],)
    a=l_tau(y-f_hat,tau=tau)
    b=l_tau(y-f_true,tau=tau)
    c = np.abs(a-b)
    return np.mean(c)  

  

def empirical_KQR_r(y,f_hat,tau):
    """function to calculate empirical excess risk for KQR without subtracting true risk"""
    y=y.reshape(y.shape[0],)
    f_hat=f_hat.reshape(f_hat.shape[0],)
    a=l_tau(y-f_hat,tau=tau)
    c = np.abs(a)
    return np.mean(c)



def KQR_estimation(K, y_train, lam, tau, truncation=False, r=None, pre_SVD=None):
    """function to accomplish estimation for kernel quantile regression"""
    n=K.shape[0]
    weight=np.ones(n)
    def full_KQR(K, y, lam):
        C = 1/(n*lam)
        G=np.vstack((np.identity(n),-np.identity(n)))
        P=matrix(K)
        q=matrix(-y_train)
        G=matrix(G)
        A=matrix(np.ones(n),(1,n))
        b=matrix([0.0])
        t_1=C*tau*weight
        t_2=C*(1-tau)*weight
        h=np.vstack((t_1.reshape(n,1),t_2.reshape(n,1)))
        h=matrix(h)
        
        solvers.options['show_progress'] = False
        result=solvers.qp(P,q,G,h,A,b, show_progress=False)
        alpha_hat=np.array(result['x'])
        y_hat=np.dot(K,alpha_hat)+result['y']
        # a=(alpha_hat,y_hat)
        return y_hat
    if truncation==False:        
        result = full_KQR(K, y_train, lam)
    else:
        if pre_SVD==None:
            K_approx = kernel_truncation(K, r)
        else:
            K_approx = kernel_truncation(K, r, pre_SVD)
        result = full_KQR(K_approx, y_train, lam)
    return result



def loss_quantile(y_train, y_est:np.ndarray, y_true:np.ndarray, loss_type:str, tau:float):
    if loss_type=="l2":
        return mean_squared_error(y_est, y_true)
    elif loss_type=="excess_risk":
        return empirical_KQR(y_train, y_est, y_true,tau=tau)

    



def choose_lam_r_quantile(K, y_train, y_true, loss_type, truncation=False, tau=0.25, )->float:
    """
    function that return the best mse under specified lambda and r
    
    parameters
    ----------
    K: array-like of shape (n_samples, n_samples)
        kernel matrix of x_train
    
    y_train: array-like of shape (n_samples, )

    lam: array-like of shape (lam_list, )
        regularization parameter, if none then choose from defalt set

    r: array-like of shape (r_list, )
        truncation parameter, if none then choose from defalt set

    truncation: bool

    return
    ------
    optimal mse
    """
    n = K.shape[0]
    lam_list = np.linspace(0.0001,1,20)
    if truncation:
        r_list = 10**np.linspace(-3, 0, 20)
        error_list = np.zeros([len(lam_list), len(r_list)])
        U, s, V = np.linalg.svd(K)
        for i in range(len(lam_list)):
            for j in range(len(r_list)):
                y_est = KQR_estimation(K, y_train, lam=lam_list[i], tau=tau, 
                                        truncation=True, r=int(r_list[j]*n), pre_SVD=(U, s, V))
                error_list[i, j] = loss_quantile(y_train, y_est, y_true, loss_type=loss_type, tau=tau)
        optimal_error = np.min(error_list)
    else:
        error_list = np.zeros(len(lam_list))
        for i in range(len(lam_list)):
            y_est = KQR_estimation(K, y_train, lam=lam_list[i], tau=tau, 
                                        truncation=False, r=None, pre_SVD=None)
            error_list[i] = loss_quantile(y_train, y_est, y_true, loss_type=loss_type, tau=tau)
        optimal_error = np.min(error_list)
    return optimal_error


def choose_lam_quantile(K, y_train, y_true, r, truncation=False, tau=0.25, loss_type="l2"):
    """
    function that return the best mse under specified lambda and r
    
    parameters
    ----------
    K: array-like of shape (n_samples, n_samples)
        kernel matrix of x_train
    
    y_train: array-like of shape (n_samples, )

    lam: array-like of shape (lam_list, )
        regularization parameter, if none then choose from defalt set

    r: array-like of shape (r_list, )
        truncation parameter, if none then choose from defalt set

    truncation: bool

    return
    ------
    optimal mse
    """
    lam_list = np.linspace(0.0001,1,20)
    mse = np.zeros(len(lam_list))
    U, s, V = np.linalg.svd(K)
    for i in range(len(lam_list)):
        y_est = KQR_estimation(K, y_train, lam=lam_list[i], tau=tau, 
                                        truncation=truncation, r=r, pre_SVD=(U, s, V))
        mse[i] = loss_quantile(y_train, y_est, y_true, loss_type, tau=tau)
    optimal_mse = np.min(mse)
    return optimal_mse


def loss_svm(y_est:np.ndarray, y_true:np.ndarray, loss_type:str):
    """function to calculate weighted one-zero or hinge loss for KSVM"""
    if loss_type =="one_zero":
        y_est = np.sign(y_est)
        y_true = np.sign(y_true)
        loss=(y_est!=y_true).mean()
    if loss_type == "l2":
        loss = mean_squared_error(y_est, y_true)
    elif loss_type=='hinge':
        y_est = np.sign(y_est)
        y_true = np.sign(y_true)
        loss=np.clip(1-y_est*y_true, a_min=0,a_max=100000).mean()
    elif loss_type=="excess_risk":
        len = y_est.shape[0]
        loss=0
        for i in range(len):
            if y_est[i]*y_true[i]<1:
                loss=loss+1-y_est[i]*y_true[i]
        loss= loss/len
    return loss  


def KSVM_estimation(K, y_train, lam, truncation=False, r=None, pre_SVD=None):
    """function to accomplish estimation for kernel SVM"""
    n_tr = K.shape[0]
    def full_KSVM(K, y_train, lam):
        C = 1/(lam*n_tr)
        K_tilde=np.dot(np.dot(np.diag(y_train),K),np.diag(y_train))
        G=np.vstack((np.identity(n_tr),-np.identity(n_tr)))

        t_1=C*np.ones(n_tr)
        t_2=np.zeros(n_tr)

        
        h=np.vstack((t_1.reshape(n_tr,1),t_2.reshape(n_tr,1)))
        
        P=matrix(K_tilde)
        q=matrix(-np.ones(n_tr))
        G=matrix(G)
        h=matrix(h)
        A=matrix(y_train,(1,n_tr),'d')
        b=matrix([0.0])

        solvers.options['show_progress'] = False
        result=solvers.qp(P,q,G,h,A,b)
        eta_hat=np.array(result['x'])
        alpha_hat=eta_hat*y_train.reshape(n_tr,1)
        y_hat=np.dot(K,alpha_hat)+result['y']
        return y_hat
    if truncation==False:
        result = full_KSVM(K, y_train, lam)
    else:
        if pre_SVD==None:
            K_approx = kernel_truncation(K, r)
        else:
            K_approx = kernel_truncation(K, r, pre_SVD)
        result = full_KSVM(K_approx, y_train, lam)
    
    result = np.array(result).reshape(-1)
    return result



def choose_lam_r_svm(K, y_train, y_true, truncation=False, loss_type="excess_risk"):
    """
    function that return the best mse under specified lambda and r
    
    parameters
    ----------
    K: array-like of shape (n_samples, n_samples)
        kernel matrix of x_train
    
    y_train: array-like of shape (n_samples, )

    lam: array-like of shape (lam_list, )
        regularization parameter, if none then choose from defalt set

    r: array-like of shape (r_list, )
        truncation parameter, if none then choose from defalt set

    truncation: bool

    return
    ------
    optimal mse
    """
    n = K.shape[0]
    lam_list = np.linspace(0.0001,1,20)
    if truncation:
        r_list = 10**np.linspace(-3, 0, 20)
        error_list = np.zeros([len(lam_list), len(r_list)])
        U, s, V = np.linalg.svd(K)
        for i in range(len(lam_list)):
            for j in range(len(r_list)):
                result = KSVM_estimation(K, y_train, lam=lam_list[i], 
                                        truncation=True, r=int(r_list[j]*n), pre_SVD=(U, s, V))
                error_list[i, j] = loss_svm(result, y_true, loss_type)
        optimal_error = np.min(error_list)
    else:
        error_list = np.zeros(len(lam_list))
        for i in range(len(lam_list)):
            result = KSVM_estimation(K, y_train, lam=lam_list[i], 
                                     truncation=False, r=None, pre_SVD=None)
            error_list[i] = loss_svm(result, y_true, loss_type)
        optimal_error = np.min(error_list)
    return optimal_error




def choose_lam_svm(K, y_train, y_true, r, truncation=False, loss_type="one_zero", pre_SVD=None):
    """
    function that return the best mse under specified lambda and r
    
    parameters
    ----------
    K: array-like of shape (n_samples, n_samples)
        kernel matrix of x_train
    
    y_train: array-like of shape (n_samples, )

    lam: array-like of shape (lam_list, )
        regularization parameter, if none then choose from defalt set

    r: array-like of shape (r_list, )
        truncation parameter, if none then choose from defalt set

    truncation: bool

    return
    ------
    optimal mse
    """
    n = K.shape[0]
    if pre_SVD==None:
        lam_list = 10**np.linspace(-3,1,5)
        mse = np.zeros(len(lam_list))
        U, s, V = np.linalg.svd(K)
        for i in range(len(lam_list)):
            y_est = KSVM_estimation(K, y_train, lam=lam_list[i], 
                                truncation=True, r=r, pre_SVD=(U, s, V))
            mse[i] = loss_svm(y_est, y_true, loss_type)
        optimal_mse = np.min(mse)
    else:
        lam_list = 10**np.linspace(-3,1,5)
        mse = np.zeros(len(lam_list))
        U, s, V = pre_SVD
        for i in range(len(lam_list)):
            y_est = KSVM_estimation(K, y_train, lam=lam_list[i], 
                                truncation=True, r=r, pre_SVD=(U, s, V))
            mse[i] = loss_svm(y_est, y_true, loss_type)
        optimal_mse = np.min(mse)
    return optimal_mse




#Kernel logistic regression (KLR)
def Newton(K, y, alpha,weight,lamb):
    """function to perform Newton–Raphson algorithm for KLR"""
    n=y.shape[0]
    p=np.exp(y*np.dot(K,alpha))/(1+np.exp(y*np.dot(K,alpha)))
    W=np.diag(p*(1-p)*weight)
    A=np.linalg.inv(np.dot(W,K)+n*lamb*np.identity(n))
    B=np.dot(np.dot(W,K),alpha)+y*(1-p)*weight
    
    return np.dot(A,B)


def KLR_estimation(K, x_train, y_train, lam, truncation=False, r=None, pre_SVD=None, T=10):
    """function to accomplish estimation for kernel logistic regression"""
    n=K.shape[0]
    W=np.ones(n)
    def full_KLR(K, y_train, lam):
        alpha=np.zeros(n)
        for t in range(T):
            alpha=Newton(K, y=y_train,weight=W,alpha=alpha, lamb=lam)
        y_hat=np.dot(K,alpha)
        # y_pre=np.zeros(x.shape[0])
        # y_pre[y_hat>=0]=1
        # y_pre[y_hat<0]=-1
        # y_pre=y_pre.reshape(x.shape[0],) 
        return y_hat
    if truncation==False:
        result = full_KLR(K, y_train, lam)
    else:
        if pre_SVD==None:
            K_approx = kernel_truncation(K, r)
        else:
            K_approx = kernel_truncation(K, r, pre_SVD)
        result = full_KLR(K_approx, y_train, lam)
    return result


def empirical_KLR(y,f_hat,f_true):
    y=y.reshape(y.shape[0],)
    f_hat=f_hat.reshape(f_hat.shape[0],)
    f_true=f_true.reshape(f_true.shape[0],)
    a=np.log(1+np.exp(-y*f_hat))
    b=np.log(1+np.exp(-y*f_true))
    c=np.abs(a-b)
    return np.mean(c)  


def loss_KLR(y_train:np.ndarray, y_est:np.ndarray, y_true:np.ndarray, loss_type:str):
    """function to calculate loss for KLR"""
    if loss_type =="one_zero":
        y_est = np.sign(y_est)
        y_true = np.sign(y_true)
        loss = (y_est!=y_true).mean()
    elif loss_type=="excess_risk":
        loss=empirical_KLR(y_train,y_est,y_true)
    return loss



def choose_lam_r_lr(K, x_train, y_train, y_true, truncation=False, loss_type="one_zero"):
    n = K.shape[0]
    lam_list = np.linspace(0.0001,1,20)
    if truncation:
        r_list = 10**np.linspace(-3, 0, 20)
        error_list = np.zeros([len(lam_list), len(r_list)])
        U, s, V = np.linalg.svd(K)
        for i in range(len(lam_list)):
            for j in range(len(r_list)):
                result = KLR_estimation(K, x_train, y_train, lam=lam_list[i], 
                                        truncation=True, r=int(r_list[j]*n), pre_SVD=(U, s, V))
                error_list[i, j] = loss_KLR(y_train, result, y_true, loss_type)
        optimal_error = np.min(error_list)
    else:
        error_list = np.zeros(len(lam_list))
        for i in range(len(lam_list)):
            result = KLR_estimation(K, x_train, y_train, lam=lam_list[i], 
                                     truncation=False, r=None, pre_SVD=None)
            error_list[i] = loss_KLR(y_train, result, y_true, loss_type)
        optimal_error = np.min(error_list)
    return optimal_error




def choose_lam_lr(K, x_train, y_train, y_true, r, truncation=False, loss_type="one_zero", pre_SVD=None):
    """
    function that return the best mse under specified lambda and r
    
    parameters
    ----------
    K: array-like of shape (n_samples, n_samples)
        kernel matrix of x_train
    
    y_train: array-like of shape (n_samples, )

    lam: array-like of shape (lam_list, )
        regularization parameter, if none then choose from defalt set

    r: array-like of shape (r_list, )
        truncation parameter, if none then choose from defalt set

    truncation: bool

    return
    ------
    optimal mse
    """

    n = K.shape[0]
    if pre_SVD==None:
        lam_list = 10**np.linspace(-3, 0,20)
        mse = np.zeros(len(lam_list))
        U, s, V = np.linalg.svd(K)
        for i in range(len(lam_list)):
            y_est = KLR_estimation(K, x_train, y_train, lam=lam_list[i], 
                                truncation=True, r=r, pre_SVD=(U, s, V))
            mse[i] = loss_KLR(y_train, y_est, y_true, loss_type)
        optimal_mse = np.min(mse)
    else:
        lam_list = 10**np.linspace(-3, 0,20)
        mse = np.zeros(len(lam_list))
        U, s, V = pre_SVD
        for i in range(len(lam_list)):
            y_est = KLR_estimation(K, x_train, y_train, lam=lam_list[i], 
                                truncation=True, r=r, pre_SVD=(U, s, V))
            mse[i] = loss_KLR(y_train, y_est, y_true, loss_type)
        optimal_mse = np.min(mse)
    return optimal_mse



#### cv
import numpy as np
from function import KQR_estimation, Kernel_rbf, choose_lam_r_quantile, choose_lam_quantile
from function import Kernel_sobo, Kernel_laplace, empirical_KQR, empirical_KQR_r
import scipy.stats as stats
import tqdm
from cvxopt import matrix, solvers
from sklearn.model_selection import KFold
import time 
from sklearn.metrics import mean_squared_error



def full_KQR(K, K_test, y_train,lam, tau):
        n = K.shape[0]
        C = 1/(n*lam)
        G=np.vstack((np.identity(n),-np.identity(n)))
        P=matrix(K)
        q=matrix(-y_train)
        G=matrix(G)
        A=matrix(np.ones(n),(1,n))
        b=matrix([0.0])
        t_1=C*tau*np.ones(n)
        t_2=C*(1-tau)*np.ones(n)
        h=np.vstack((t_1.reshape(n,1),t_2.reshape(n,1)))
        h=matrix(h)
        
        solvers.options['show_progress'] = False
        result=solvers.qp(P,q,G,h,A,b, show_progress=False)
        alpha_hat=np.array(result['x'])
        y_hat=np.dot(K_test,alpha_hat)+result['y']
        # a=(alpha_hat,y_hat)
        return y_hat


def test_error(K, train_sample,y_train, test_sample, y_test, r, lam,  tau, loss_type):
    n = train_sample.shape[0]
    m = test_sample.shape[0]
    K_train = K(train_sample, train_sample)
    K1 = K(test_sample, train_sample)
    U, S, V = np.linalg.svd(K_train)
    psi_alpha = np.zeros([r, n])
    pin1 = time.time()
    for i in range(r):
        psi_alpha[i, :] = np.linalg.solve(K_train + 10**(-6)*np.eye(n), U[:, i])
    K_test = K1@psi_alpha.T@np.diag(S[:r])@psi_alpha@K_train
    K_train_t = K_train@psi_alpha.T@np.diag(S[:r])@psi_alpha@K_train
    pin2 = time.time()
    # print("time for psi_alpha", pin2-pin1)
    y_hat = full_KQR(K_train_t, K_test, y_train, lam, tau)
    if loss_type == "l2":
        return mean_squared_error(y_hat, y_test)
    elif loss_type == "excess_risk":
        return empirical_KQR_r(y_test, y_hat,tau=tau)
    else:
        raise ValueError("loss_type should be l2, excess_risk, l1 or quantile")
    


def cv(K, sample_x, sample_y, tau, loss_type):
    lam_list = 10**np.linspace(-4, 0, 10)
    r_list = 10**np.linspace(-3, 0, 10)
    kf = KFold(n_splits=5, shuffle=True)
    mse_min = np.inf
    lam_index = 0
    r_index = 0
    for i in range(r_list.shape[0]):
        for j in range(lam_list.shape[0]):
            mse_sum = 0
            for train_index, test_index in kf.split(sample_x):
                train_sample = sample_x[train_index]
                len_train = train_sample.shape[0]
                test_sample = sample_x[test_index]
                y_train = sample_y[train_index]
                y_test = sample_y[test_index]
                r = max(int(r_list[i]*len_train), 1)
                mse = test_error(K, train_sample, y_train, test_sample, y_test, r , lam_list[j], tau, loss_type)
                mse_sum += mse
            mse = mse_sum/5
            if mse < mse_min:
                mse_min = mse
                lam_index = j
                r_index = i
    # print("cv of TKM choose best r=", r_list[r_index], "best lam=", lam_list[lam_index], "min mse=", mse_min)
    return r_list[r_index], lam_list[lam_index], mse_min



def truncation_Ksvm(K, sample_x, sample_y, r, lam, loss_type, tau, f_true):
    n = sample_x.shape[0]
    K_train = K(sample_x, sample_x)
    test_sample = np.random.rand(1000)
    U, S, V = np.linalg.svd(K_train)
    psi_alpha = np.zeros([r, n])
    for i in range(r):
        psi_alpha[i, :] = np.linalg.solve(K_train + 10**(-6)*np.eye(n), U[:, i])
    y_true = f_true(test_sample)
    test_y = y_true + np.random.normal(0, 2, 1000) - stats.norm.ppf(tau, loc=0, scale=sd)
    K_train_t = K_train@psi_alpha.T@np.diag(S[:r])@psi_alpha@K_train
    K1 = K(test_sample, sample_x)
    K_test = K1@psi_alpha.T@np.diag(S[:r])@psi_alpha@K_train
    y_hat = full_KQR(K_train_t,K_test, sample_y, lam, tau)
    if loss_type == "l2":
        return mean_squared_error(y_hat, y_true)
    elif loss_type == "excess_risk":
        return empirical_KQR(test_y, y_hat, y_true, tau)
    else:
        raise ValueError("loss_type should be l2, excess_risk, l1 or quantile")





# now we derive the mse of kernel regerssion

def test_error_full_kernel(K, train_sample, y_train, test_sample, y_test, lam, tau, loss_type):
    n = train_sample.shape[0]
    m = test_sample.shape[0]
    K_train = K(train_sample, train_sample)
    K1 = K(test_sample, train_sample)
    y_hat = full_KQR(K_train, K1, y_train, lam, tau)
    if loss_type == "l2":
        return mean_squared_error(y_hat, y_test)
    elif loss_type == "excess_risk":
        return empirical_KQR_r(y_test, y_hat,tau=tau)
    else:
        raise ValueError("loss_type should be l2, excess_risk, l1 or quantile")


def cv_full_kernel(K, sample_x, sample_y, tau, loss_type):
    lam_list = 10**np.linspace(-4, 0, 10)
    kf = KFold(n_splits=5, shuffle=True)
    mse_min = np.inf
    lam_index = 0
    for j in range(len(lam_list)):
        mse_sum = 0
        for train_index, test_index in kf.split(sample_x):
            train_sample = sample_x[train_index]
            len_train = train_sample.shape[0]
            test_sample = sample_x[test_index]
            y_train = sample_y[train_index]
            y_test = sample_y[test_index]
            # print("r=", r)
            mse = test_error_full_kernel(K, train_sample, y_train, test_sample, y_test, lam_list[j], tau, loss_type)
            mse_sum += mse
        mse = mse_sum/5
        if mse < mse_min:
            mse_min = mse
            lam_index = j
    # print( "In full kernel regression best lam=", lam_list[lam_index], "min mse=", mse_min)
    return  lam_list[lam_index], mse_min




def Ksvm(K, sample_x, sample_y, lam, loss_type, tau, f_true):
    n = sample_x.shape[0]
    K_train = K(sample_x, sample_x)
    test_sample = np.random.rand(1000)
    y_true = f_true(test_sample)
    test_y = y_true + np.random.normal(0, 2, 1000) - stats.norm.ppf(tau, loc=0, scale=sd)
    K1 = K(test_sample, sample_x)
    y_hat = full_KQR(K_train, K1, sample_y, lam, tau)
    if loss_type == "l2":
        return mean_squared_error(y_hat, y_true)
    elif loss_type == "excess_risk":
        return empirical_KQR(test_y, y_hat, y_true,tau=tau)
    else:
        raise ValueError("loss_type should be l2, excess_risk, l1 or quantile")



