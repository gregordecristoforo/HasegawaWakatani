Code used for my master thesis to simulate the interchange instability in magnetized plasmas. The code will feature:
* Biperiodic domain (perpendicular to $\textbf{B}$) ✓
* Fast Fourier transform for spatial derivatives (FFTW) ✓
* Third order stiffly stable time integrator ✓
* HDF data output for binary format storage X
* 2/3 Antialiasing on quadratic terms X
* Diagnostic modules ✓

![Alt Text](vorticity.gif)

The code atempts to be modular and generalizable to be able to solve other spectral problems
