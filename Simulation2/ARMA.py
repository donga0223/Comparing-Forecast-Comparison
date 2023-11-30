import numpy as np
import random

def ARMA_gen(heterotype, timeseries_type, theta, replicate, sample_size = 52):
    random.seed(replicate)
    if heterotype == "hetero":
        mean1 = np.repeat(0,sample_size)
        cov1 = np.diag(np.random.exponential(1,sample_size))
        u1 = np.random.multivariate_normal(mean1, cov1, 1)
        u = u1[0,:]
    elif heterotype == "nohetero":
        u = np.random.normal(0,1,sample_size)

    if timeseries_type == "MA":
        deno = np.sqrt(1+theta**2)
        aa = np.roll(u,-1)
        aa[(sample_size-1)] = 0
        nume = u + theta*aa
        #e = nume / deno
        e = nume 
    elif timeseries_type == "AR":
        e = np.empty([sample_size])
        e0 = np.random.normal(0,1,1)
        for i in range(sample_size):
            if i == 0:
                e[0] = theta*e0+u[0]
            else:
                e[i] = theta*e[(i-1)]+u[i]
    e = e.reshape(sample_size,1,1)
    return e



def ARMA_gen_byvar_acf(heterotype, timeseries_type, marginal_var, acf, replicate, sample_size = 52, intercept = 0):
    random.seed(replicate)
    if heterotype == "hetero":
        mean1 = np.repeat(0,sample_size)
        cov1 = np.diag(np.random.exponential(1,sample_size))
        u1 = np.random.multivariate_normal(mean1, cov1, 1)
        u = u1[0,:]
    elif heterotype == "nohetero":
        u = np.random.normal(0,1,sample_size)

    if timeseries_type == "MA":
        if acf == 0:
            theta = 0
        else :
            theta = (1 - np.sqrt(1-4*acf**2))/(2*acf)
        deno = np.sqrt(1+theta**2)
        aa = np.roll(u,-1)
        aa[(sample_size-1)] = 0
        nume = u + theta*aa
        if marginal_var == "True":
            e = nume / deno
        elif marginal_var == "False":
            e = nume
        elif marginal_var == "2times":
            e = 2*(nume / deno)
    elif timeseries_type == "AR":
        theta = acf
        normalize = np.sqrt(1-theta**2)
        e = np.empty([sample_size])
        e0 = np.random.normal(0,1,1)
        for i in range(sample_size):
            if i == 0:
                e[0] = theta*e0+u[0]
            else:
                e[i] = theta*e[(i-1)]+u[i]
        if marginal_var == "True":
            e = e*normalize
        elif marginal_var == "False":
            e = e
        elif marginal_var == "2times":
            e = 2*e*normalize

    elif timeseries_type == "MA5_same":
        if acf < 0.8:
            theta = (1-np.sqrt(1-4*acf*(5*acf-4)))/(2*(5*acf-4))
        elif acf == 0.8 :
            theta = 0.8
        deno = np.sqrt(1+5*theta**2)
        a1 = np.roll(u,-1)
        a2 = np.roll(u,-2)
        a3 = np.roll(u,-3)
        a4 = np.roll(u,-4)
        a5 = np.roll(u,-5)
        a1[(sample_size-1)] = 0
        a2[(sample_size-2):] = 0
        a3[(sample_size-3):] = 0
        a4[(sample_size-4):] = 0
        a5[(sample_size-5):] = 0
        nume = u + theta*a1 + theta*a2 + theta*a3 + theta*a4 + theta*a5
        if marginal_var == "True":
            e = nume / deno
        elif marginal_var == "False":
            e = nume
        elif marginal_var == "2times":
            e = 2*(nume / deno)
            
    elif timeseries_type == "MA5_exp":
        if acf < 0.7:
            theta = acf
        elif acf == 0.7 :
            theta = 0.71
        elif acf == 0.8 :
            theta = 0.86
        deno = np.sqrt(1+theta**2+theta**4+theta**6+theta**8+theta**10)
        a1 = np.roll(u,-1)
        a2 = np.roll(u,-2)
        a3 = np.roll(u,-3)
        a4 = np.roll(u,-4)
        a5 = np.roll(u,-5)
        a1[(sample_size-1)] = 0
        a2[(sample_size-2):] = 0
        a3[(sample_size-3):] = 0
        a4[(sample_size-4):] = 0
        a5[(sample_size-5):] = 0
        nume = u + a1*theta + a2*theta**3 + a3*theta**5 + a4*theta**7 + a5*theta**7
        if marginal_var == "True":
            e = nume / deno
        elif marginal_var == "False":
            e = nume
        elif marginal_var == "2times":
            e = 2*(nume / deno)
    e = e + intercept
    e = e.reshape(sample_size,1,1)
    return e
