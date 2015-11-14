import numpy as np
import numpy.random as npr
npr.seed(0)
import matplotlib.pyplot as plt
plt.rc('font', family='serif')


def annealed_importance_sampling(draw_exact_initial_sample,
                                 transition_kernels,
                                 annealing_distributions,
                                 n_samples=1000):
    '''
    draw_exact_initial_sample:
        Signature:
            Arguments: none
            Returns: R^d

    transition_kernels:
        length-T list of functions, each function signature:
            Arguments: R^d
            Returns: R^d

        can be any transition operator that preserves its corresponding annealing distribution

    annealing_distributions:
        length-T list of functions, each function signature:
            Arguments: R^d
            Returns: R^+

        annealing_distributions[0] is the initial density
        annealing_distributions[-1] is the target density

    n_samples:
        positive integer

    '''

    dim=len(draw_exact_initial_sample())
    T = len(annealing_distributions)
    weights = np.ones(n_samples,dtype=np.double)
    ratios = []

    xs = []
    for k in range(n_samples):
        x = np.zeros((T,dim))
        ratios_ = np.zeros(T-1,dtype=np.double)
        x[0] = draw_exact_initial_sample()

        for t in range(1,T):


            f_tminus1 = annealing_distributions[t-1](x[t-1])
            f_t = annealing_distributions[t](x[t-1])

            ratios_[t-1] = f_t/f_tminus1
            weights[k] *= ratios_[t-1]

            x[t] = transition_kernels[t](x[t-1],target_f=annealing_distributions[t])

        xs.append(x)
        ratios.append(ratios_)

    estimated_Z_ratio = (np.cumsum(weights)/np.arange(1,len(weights)+1))

    return estimated_Z_ratio,np.array(xs), weights, np.array(ratios)


if __name__=='__main__':
    ''' run simple example '''

    # define initial, target, and intermediate distributions
    num_intermediates = 10
    betas = np.linspace(0,1,num_intermediates+2)
    dim=1

    def initial_density(x):
        return np.exp(-((x)**2).sum()/2)

    def draw_from_initial():
        return npr.randn(dim)

    def target_density(x):
        return np.exp(-((x-3)**2).sum()/2)

    Z_ratio = 1.0 # ratio of normalizing constants of initial and target

    # define annealing distributions
    from annealing_distributions import GeometricMean
    annealing_distributions = [GeometricMean(initial_density,target_density,beta) for beta in betas]

    # define transition kernels
    from transition_kernels import gaussian_random_walk
    transition_kernels = [gaussian_random_walk]*len(annealing_distributions)

    # run AIS
    Z_hat, xs, weights, ratios = annealed_importance_sampling(draw_from_initial,
                                 transition_kernels,
                                 annealing_distributions,
                                 n_samples=10000)

    # plot results
    plt.plot(Z_hat)
    plt.hlines(Z_ratio,0,len(weights))
    plt.xlabel('# samples')
    plt.ylabel(r'Estimated $\mathcal{Z}_T / \mathcal{Z}_1$')
    plt.title(r'Estimated $\mathcal{Z}_T / \mathcal{Z}_1$')
    plt.ylim(0,3.0)
    plt.show()
