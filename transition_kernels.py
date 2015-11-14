## TRANSITION KERNELS

import numpy.random as npr

def gaussian_random_walk(x,
                      target_f,
                      n_steps=10,
                      scale=0.5):
    ''' random walk metropolis-hastings with spherical gaussian
    proposals '''

    x_old = x
    f_old = target_f(x_old)
    dim=len(x)

    for i in range(n_steps):

        proposal = x_old + npr.randn(dim)*scale
        f_prop = target_f(proposal)

        if (f_prop / f_old) > npr.rand():
            x_old = proposal
            f_old = f_prop

    return x_old
