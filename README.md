Code used for my master thesis to simulate the interchange instability in magnetized plasmas, described by the following equations

$$\frac{\partial n}{\partial t} + \\{\phi, n\\} - gn\frac{\partial\phi}{\partial y} + g\frac{\partial n}{\partial y} = D_n\nabla^2_\perp n - \sigma_nn\exp(\phi)$$

$$\frac{\partial\Omega}{\partial t} + \\{\phi,\Omega\\} + g\frac{\partial\ln(n)}{\partial y} = D_\Omega\nabla^2_\perp\Omega + \sigma_\Omega[1-\exp(\phi)]$$

with $\Omega = \nabla^2\phi$. The code will feature:
* Biperiodic domain (perpendicular to $\textbf{B}$) ✓
* Fast Fourier transform for spatial derivatives (FFTW) ✓
* Third order stiffly stable time integrator ✓
* HDF data output for binary format storage X
* 2/3 Antialiasing on quadratic terms X
* Diagnostic modules ✓

![Alt Text](vorticity.gif)

The code atempts to be modular and generalizable to be able to solve other spectral problems
