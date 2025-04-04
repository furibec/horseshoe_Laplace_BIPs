# Horseshoe prior and Laplace approximation for Bayesian inverse problems in imaging
This repository contains python implementations of the main methods proposed in these articles:

1. F. Uribe, Y. Dong, P. C. Hansen (2023): [Horseshoe priors for edge-preserving linear Bayesian inversion](https://epubs.siam.org/doi/10.1137/22M1510364). SIAM Journal on Scientific Computing. 45(3), B337-B365.
2. F. Uribe, J. M. Bardsley, Y. Dong, P. C. Hansen, N. A. B. Riis (2022): [A hybrid Gibbs sampler for edge-preserving tomographic reconstruction with uncertain view angles](https://epubs.siam.org/doi/10.1137/21M1412268). SIAM/ASA Journal on Uncertainty Quantification. 10(3), 1293-1320.

Both approaches rely on the Gibbs sampler. For the second paper, the uncertain view angles are omited, and the focus is on setting up an edge-preserving prior based on the Laplace approximation of the Laplace prior.

Any suggestions, corrections or improvements are kindly accepted :-)
