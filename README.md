# Parameters-Optimization
This repository performs parameter estimation & optimization of dynamics systems by solving least-squares problems with numerical optimization methods.
# Define the Cost Function

$min \hspace{0.5em} \Sigma (r_{k}-x_{k}(\mathbf{\theta}))^2$\
s.t.
$\mathbf{\dot{x}}=\mathbf{f}(\mathbf{x},\mathbf{\theta} )$
# Calculate the Gradient
There are two approaches to compute the gradient.
1. Pontryagin's adjoint method\
   Compute the gradient by solving the adjoint differential equations. This method provides accurate and efficient gradient estimation; however, deriving the adjoint equations is not easy for general dynamics systems.
2. Numerical Differentiation\
   Using finite difference method to estimate the gradient. This approach is easy to implement, but as the number of parameters grows,
   the computational cost will become unacceptable, and since the numerical differentiation is quite sensitive to noise, using the finite difference method      directly will make Quasi-Newton methods fail to converge in some ill-condition problems.
# Noisy Gradient Problems
* Some stochastic optimization methods can handle the noisy gradient such as SGDM and Adam, which are two prevalent optimization methods
in the deep learning community.
* If the gradient estimation is accurate, the Quasi-Newton method will be much superior to first-order methods.
