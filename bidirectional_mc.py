import numpy as np

from ais import annealed_importance_sampling


def bidirectional_ais(draw_exact_initial_sample,
                      draw_exact_target_sample,
                      transition_kernels,
                      annealing_distributions,
                      n_samples=10000):
    '''
    Run AIS forward and backward to get stochastic upper and lower bounds
    on the log ratio of normalizing constants between initial and target
    distributions.

    Reference:
    Sandwiching the marginal likelihood using bidirectional Monte Carlo
    (Roger B. Grosse, Zoubin Ghahramani, Ryan P. Adams, 2015)
    http://arxiv.org/abs/1511.02543

    draw_exact_initial_sample:
        Signature:
            Arguments: none
            Returns: R^d

    draw_exact_target_sample:
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

    z_f, xs_f, weights_f, ratios_f = annealed_importance_sampling(draw_exact_initial_sample,
                                     transition_kernels,
                                     annealing_distributions,
                                     n_samples=n_samples)

    forward_results = z_f, xs_f, weights_f, ratios_f

    z_r, xs_r, weights_r, ratios_r = annealed_importance_sampling(draw_exact_target_sample,
                                 transition_kernels[::-1],
                                 annealing_distributions[::-1],
                                 n_samples=n_samples)

    reverse_results = z_r, xs_r, weights_r, ratios_r

    # process results into estimates of log(Z_T / Z_1)
    stoch_upper_bound = np.log(z_f)
    stoch_lower_bound = np.log(1/z_r)

    return stoch_upper_bound,stoch_lower_bound,forward_results,reverse_results


if __name__=='__main__':
    import numpy.random as npr
    npr.seed(0)
    import matplotlib.pyplot as plt
    plt.rc('font', family='serif')

    # define initial, target, and intermediate distributions
    num_intermediates = 10
    betas = np.linspace(0,1,num_intermediates+2)
    dim=1

    # initial
    def initial_density(x):
        return np.exp(-((x)**2).sum()/2)

    def draw_from_initial():
        return npr.randn(dim)

    # target
    def target_density(x):
        return np.exp(-((x-3)**2).sum()/2)

    def draw_from_target():
        return npr.randn(dim)+3

    Z_ratio = 1.0 # ratio of normalizing constants of initial and target

    # define annealing distributions
    from annealing_distributions import GeometricMean
    annealing_distributions = [GeometricMean(initial_density,target_density,beta) for beta in betas]

    # define transition kernels
    from transition_kernels import gaussian_random_walk
    transition_kernels = [gaussian_random_walk]*len(annealing_distributions)

    # run bidirectional_monte_carlo
    stoch_upper_bound,stoch_lower_bound,_,_ = bidirectional_ais(draw_from_initial,
        draw_from_target,
        transition_kernels,
        annealing_distributions,
        n_samples=10000)

    # plot results
    plt.plot(stoch_upper_bound,label='Stochastic upper bound')
    plt.plot(stoch_lower_bound,label='Stochastic lower bound')
    plt.hlines(np.log(Z_ratio),0,len(stoch_upper_bound))
    plt.xlabel('# samples')
    plt.ylabel(r'Estimated $\log ( \mathcal{Z}_T / \mathcal{Z}_1)$')
    plt.title(r'Estimated $\log ( \mathcal{Z}_T / \mathcal{Z}_1)$')
    plt.legend(loc='best')
    plt.show()
