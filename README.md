# Parameters-Optimization
Implementation of parameters estimation and optimization algorithms.
# Define the Cost Function
This repository performs parameter estimation & optimization of dynamics systems by solving least-squares problems with numerical optimization methods, i.e.\
$min \hspace{0.5em} \Sigma (r_{k}-x_{k}(\mathbf{\theta}))^2$\
s.t.
$\mathbf{\dot{x}}=\mathbf{f(\mathbf{x},\mathbf{\theta} )}$
# Calculate the Gradient
There are two approaches to compute the gradient.
1. Pontryagin's adjoint method\
   Compute the gradient by solving the co-state differential equations. This method provides accurate gradient estimation, however
   , derive the co-state equation is not an easy task for general dynamics systems.
2. Numerical Differentiation\
   Using finite difference method to estimate the gradient directly. This approach is easy to implement, but as the number of parameters grows,
   the computational cost will become unacceptable, and since the numerical differentiation is quite sensitive to noise, using the gradient
   estimated by the finite     difference method directly will make optimization fail to converge in some ill-condition problems.
# Noisy Gradient Problems
* Some stochastic optimization methods can handle the noisy gradient such as SGDM and Adam, which are two prevalent optimization methods
in the deep learning community.
* If the gradient estimation is accurate, the Quasi-Newton method will be much superior to first-order methods.
