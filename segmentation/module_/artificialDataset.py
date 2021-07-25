import numpy as np

def createJumpingMean():
    samples=[]

    # sigma
    sigma = 0.5
    # mu
    # noiseMeanList=[]
    # for i in range(10):
    #     if i==0:
    #         noiseMeanList.append(0)
    #     else:
    #         nextMean=noiseMeanList[-1]+(i+1)/16
    #         noiseMeanList.append(nextMean)
    
    mean = 0
    for i in range(10):
        # noise=np.random.normal(noiseMeanList[i], sigma, 100)
        noise=np.random.normal(mean, sigma, 100)
        for j in range(100):
            if i==0 and j<2:
                samples.append(0)
            else:
                sample = 0.6*samples[-1] - 0.5*samples[-2] + noise[j]
                samples.append(sample)
        mean+=2

    return samples

def createScalingVariance():
    samples=[]

    # noiseMeanList=[]
    # for i in range(50):
    #     if i==0:
    #         noiseMeanList.append(0)
    #     else:
    #         nextMean=noiseMeanList[-1]+(i+1)/16
    #         noiseMeanList.append(nextMean)
    
    # sigmaList=[]
    # for i in range(1, 51):
    #     if i%2==1:
    #         sigmaList.append(1)
    #     else:
    #         sigmaList.append(np.log(np.exp(1)+i/4))
    
    
    for i in range(10):
        sigma=np.random.uniform(0.01, 1)
        # noise=np.random.normal(0, sigmaList[i], 100)
        noise=np.random.uniform(0, sigma, 100)
        for j in range(100):
            if i==0 and j<2:
                samples.append(0)
            else:
                sample = 0.6*samples[-1] - 0.5*samples[-2] + noise[j]
                samples.append(sample)
    
    return samples

def createSwitchingCovariance():
    samples=[]

    x=np.zeros(5000)
    y=np.zeros(5000)

    mean = np.array([0,0])
    for i in range(1, 51):
        if i%2==1:
            cov=np.matrix([[1, -4/5-(i-2)/500], [-4/5-(i-2)/500, 1]])
        else:
            cov=np.matrix([[1, 4/5+(i-2)/500], [4/5+(i-2)/500, 1]])

        data=np.random.multivariate_normal(mean, cov, 100).T
        for j in range(100):
            x[(i-1)*100+j]=data[0,j]
            y[(i-1)*100+j]=data[1,j]

    samples.append(x)
    samples.append(y)

    return samples

def createChangingFrequency():
    samples=[]

    currweight = prevweight = 0

    for i in range(10):
        if i==0:
            currweight=1
        else:
            # currweight=prevweight*np.log(np.exp(1)+(i+1)/2)
            currweight*=5
        noise=np.random.normal(0, 0.8, 100)
        for j in range(100):
            t=i*100+(j+1)
            sample=np.sin(currweight*t)+noise[j]
            samples.append(sample)

    return samples
