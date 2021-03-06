# Contact_SymplecticIntegrator
Code to reproduce results in "Fast symplectic integrator for Nesterov-type acceleration method"

MNIST datasets are too large hence X datasets are ziped. Please unzip before running.

To reproduce the results on the comparison of SI2 and RK2/4, run

> % julia compare_SI2_RK24.jl

Pdf files showing the convergence behavior with different convergence rate parameter values of SI2, RK2, and RK4 on five datasets will be generated in the current directly.

To reproduce the results on the comparison of SI2 to NAG, run

> % julia compare_SI2_NAG.jl

Pdf files showing the convergence behavior of SI2 and NAG on five datasets will be generated in the current directly.

Finally, to evaluate the computational time of each method, run

> % julia timing.jl

It will take several minutes to repeatedly run minimization experiments ten times for each method/dataset.

