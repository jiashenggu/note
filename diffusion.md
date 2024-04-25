# Playground v2.5

## EDM
EDM brings two distinct advantages: (1) Like Zero Terminal SNR, the EDM noise schedule exhibits
a near-zero signal-to-noise ratio for the final “timestep”. This removes the need for Offset Noise and
fixes muted colors. (2) EDM takes a first-principles approach to designing the training and sampling
processes, as well as preconditioning of the UNet. This enables the EDM authors to make clear
design choices that lead to better image quality and faster model convergence.

## Balanced bucket sampling strategy
While we followed a bucketing strategy similar to SDXL’s, we
carefully crafted the data pipeline to ensure a more balanced bucket sampling strategy across various
aspect ratios. Our strategy avoids catastrophic forgetting and helps the model not be biased towards
one ratio or another.

# [测试函数法推导连续性方程和Fokker-Planck方程](https://kexue.fm/archives/9461)
(19)
In summary, the absence of the f(x) in the second-order term is due to the specific rules of Itō's calculus that account for the quadratic variation of the Brownian motion. The f(x) term is not ignored; it's that the second-order term in the stochastic Taylor expansion is specifically related to the variance of the stochastic process, which involves g(x) rather than 
 f(x).
