# GSoC 2022 - Expanding FEniCSx electromagnetic demos

In this website, I will provide an overview of my work as a GSoC 2022 contributor for
the [FEniCS project](https://fenicsproject.org/) (sponsored by [NumFOCUS](https://numfocus.org/))
and its main environment [DOLFINx](https://github.com/FEniCS/dolfinx).
The goal of the project was to expand DOLFINx electromagnetic demos, in order to
promote the use of the FEniCSx environment for solving electromagnetic and photonic
problems. During GSoC 2022, I have implemented demos showing how to:

- implement scattering boundary conditions;
- implement perfectly matched layers;
- solve electromagnetic eigenvalue problems;
- solve 3D electromagnetic problems for axisymmetric structures.

All the problems are solved within the time-harmonic approximation, and use simple
geometries, but can be easily generalized to more complicated study cases.
The documentation for the demos has been provided under the form of Python
comments that can be visualized as text and math formula in [Jupyter notebook](https://jupyter.org/)
by using [Jupytext](https://jupytext.readthedocs.io/en/latest/install.html). Besides,
each demo contains tests for comparing the DOLFINx outputs with analytical results
for each problem.

This website has been built with Jupyterbook, and on the left-hand side you can find
the Jupyter notebook of the demos I have written. In this way, you can directly
visualize the mathematical documentation in the demos, the code, the outputs, and
you can play with them by using Binder (just click on the <i class="fa fa-rocket"></i> icon).

With respect to the original plan, we have not developed the demo showing how to
handle complex numbers in DOLFINx, and the demo showing how to use the `MPI.COMM_SELF`
communicator. In the first case, the demo was then considered unnecessary, since all the
main features of complex numbers in DOLFINx naturally arises in all the developed electromagnetic demos, that
widely implement them. In the second case, the demo has been substituted with the waveguide
demo, which was not planned at first. However, I plan to work on the `MPI.COMM_SELF` demo
in the weeks following the end of GSoC 2022.

In my opinion, the original goal has been successfully reached. Indeed, even if many pull
requests are still open, the main work for the demos have been done, and they just need
to pass the final rounds of reviews by FEniCSx reviewers.

## List of contributions

### Pull Requests

- [PR #2237](https://github.com/FEniCS/dolfinx/pull/2237) (**open**): this pull request
adds the demo showing how to set a time-harmonic electromagnetic problem
with scattering boundary conditions:
  - `demo_scattering_boundary_conditions.py`: solves the time-harmonic problem
  of a TM-polarized plane wave scattered by a gold wire using
  scattering boundary conditions;
  - `mesh_wire.py`: generates the mesh for the demo;
  - `analytical_efficiencies_wire.py`: calculates the analytical efficiencies
  for a wire;
- [PR #2276](https://github.com/FEniCS/dolfinx/pull/2276) (**open**): this pull request
adds the demo showing how to set a time-harmonic electromagnetic problem with perfectly matched layers:
  - `demo_pml.py`: demo for solving the time-harmonic problem of a TM-polarized plane wave scattered by a gold wire using perfectly matched layers;
  - `mesh_wire_pml.py`: generates the mesh for the demo;
  - `analytical_efficiencies_wire.py`: calculates the analytical efficiencies
  for a wire;
- [PR #2338](https://github.com/FEniCS/dolfinx/pull/2338) (**open**): this pull request adds
the demo showing how to solve a time-harmonic electromagnetic eigenvalue problem problem with DOLFINx and SLEPc:
  - `demo_waveguide.py`: solves the eigenvalue problem associated with an electromagnetic half-loaded
  waveguide with SLEPc;
  - `analytical_modes.py`: verifies if FEniCSx modes satisfy the analytical equations for the
  half-loaded waveguide;
- [PR #2339](https://github.com/FEniCS/dolfinx/pull/2339) (**open**): this pull request adds the demo
showing how to solve a time-harmonic electromagnetic problem for axisymmetric geometry:
  - `demo_axis.py`: solves the time-harmonic problem of a plane wave scattered by a sphere
  within the axisymmetric approximation, using perfectly matched layers
  - `mesh_sphere_axis.py`: generates the mesh for the demo
- [PR #2357](https://github.com/FEniCS/dolfinx/pull/2357) (**merged**): this pull request adds the line
`from dolfinx.io import gmshio` in `python/io/__init__.py` so that `gmshio` is considered a module
in DOLFINx. Besides, it also removes an unnecessary `gmshio` string in the `has_adios2` conditional
block.

### Issues

- [GH issue #2343](https://github.com/FEniCS/dolfinx/issues/2343) (**closed as completed**): issue
showing an inconsistency when solving problems with `MixedElement` having `Lagrange` elements. This
issue arose during the writing for [PR #2339](https://github.com/FEniCS/dolfinx/pull/2339), when
I noticed different DOLFINx outputs when changing the `degree` of `Lagrange` elements. The root of
this issue was a wrong permutation inside `MixedElement`, which has been then fixed with [PR #2347](https://github.com/FEniCS/dolfinx/pull/2347).

### What's next

The plan after the end of the Google Summer of Code is the following one:

- Work on merging the open pull requests;
- Develop a demo showing how to use the `MPI.COMM_SELF` communicator; this will be
particularly useful when solving parameterized problem as in the axisymmetric case, where
`MPI.COMM_SELF` would allow us to split the multiple harmonic numbers over multiple processors;
- Animate DOLFINx solutions with PyVista;
- Project axisymmetric solutions in 3D using PyVista (e.g. by using [extrude rotation](https://docs.pyvista.org/examples/01-filter/extrude-rotate.html));
- Develop more complicated demos (e.g. demos involving periodic boundary conditions);
- Join more discussions on [discourse](https://fenicsproject.discourse.group/).

## Highlights of the demos

### Demo #1: Scattering Boundary Conditions

This demo shows how to implement scattering boundary conditions for a time-harmonic
electromagnetic problem. In particular, we show how to use scattering boundary
conditions to calculate the scattering of a TM-polarized plane wave from an infinite
gold wire. The demo can also be considered an introductory demo for
electromagnetic problems, since it also shows how to derive the weak
form of the corresponding Maxwell's equations. Indeed, we start from
these equations (Maxwell's equations + scattering boundary conditions):

$$
-\nabla \times \nabla \times \mathbf{E}_s+\varepsilon_{r} k_{0}^{2}
\mathbf{E}_s
+k_{0}^{2}\left(\varepsilon_{r}-\varepsilon_{b}\right)
\mathbf{E}_{\mathrm{b}}=0 \textrm{ in } \Omega
$$

$$\mathbf{n} \times
\nabla \times \mathbf{E}_s+\left(j k_{0}n_b + \frac{1}{2r}
\right) \mathbf{n} \times \mathbf{E}_s
\times \mathbf{n}=0\quad \textrm{ on } \partial \Omega
$$

and arrive at this weak form:

$$
\begin{align}
& \int_{\Omega}-(\nabla \times \mathbf{E}_s) \cdot (\nabla \times
\bar{\mathbf{v}})+\varepsilon_{r} k_{0}^{2} \mathbf{E}_s \cdot
\bar{\mathbf{v}}+k_{0}^{2}\left(\varepsilon_{r}-\varepsilon_b\right)
\mathbf{E}_b \cdot \bar{\mathbf{v}}~\mathrm{d}x \\ +&\int_{\partial \Omega}
\left(j n_bk_{0}+\frac{1}{2r}\right)( \mathbf{n} \times \mathbf{E}_s \times
\mathbf{n}) \cdot \bar{\mathbf{v}} ~\mathrm{d} s = 0.
\end{align}
$$

which we implement in DOLFINx in the following way:

```
F = - ufl.inner(ufl.curl(Es), ufl.curl(v)) * dDom \
    + eps * (k0**2) * ufl.inner(Es, v) * dDom \
    + (k0**2) * (eps - eps_bkg) * ufl.inner(Eb, v) * dDom \
    + (1j * k0 * n_bkg + 1 / (2 * r)) \
    * ufl.inner(ufl.cross(Es_3d, n_3d), ufl.cross(v_3d, n_3d)) * dsbc
```

Besides, we also show how to implement the background field $ \mathbf{E}_b = -\sin\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_xv+ \cos\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_y$:

```
class BackgroundElectricField:

    def __init__(self, theta, n_b, k0):
        self.theta = theta
        self.k0 = k0
        self.n_b = n_b

    def eval(self, x):

        kx = self.n_b * self.k0 * np.cos(self.theta)
        ky = self.n_b * self.k0 * np.sin(self.theta)
        phi = kx * x[0] + ky * x[1]

        ax = np.sin(self.theta)
        ay = np.cos(self.theta)

        return (-ax * np.exp(1j * phi), ay * np.exp(1j * phi))
```

and how to calculate the efficiencies:
$$
\begin{align}

& Q_{abs} = \operatorname{Re}\left(\int_{\Omega_{m}} \frac{1}{2}

\frac{\operatorname{Im}(\varepsilon_m)k_0}{Z_0n_b}

\mathbf{E}\cdot\hat{\mathbf{E}}dx\right) \\

& Q_{sca} = \operatorname{Re}\left(\int_{\partial\Omega} \frac{1}{2}

\left(\mathbf{E}_s\times\bar{\mathbf{H}}_s\right)

\cdot\mathbf{n}ds\right)\\ \\

& Q_{ext} = Q_{abs} + Q_{sca}, \\

& q_{abs} = \frac{Q_{abs}}{I_0\sigma_{gcs}} \\

& q_{sca} = \frac{Q_{sca}}{I_0\sigma_{gcs}} \\

& q_{ext} = q_{abs} + q_{sca}, \\

\end{align}
$$

```
Z0 = np.sqrt(mu_0 / epsilon_0)

# Magnetic field H
Hsh_3d = -1j * curl_2d(Esh) / (Z0 * k0 * n_bkg)

Esh_3d = ufl.as_vector((Esh[0], Esh[1], 0))
E_3d = ufl.as_vector((E[0], E[1], 0))

# Intensity of the electromagnetic fields I0 = 0.5*E0**2/Z0
# E0 = np.sqrt(ax**2 + ay**2) = 1, see background_electric_field
I0 = 0.5 / Z0

# Geometrical cross section of the wire
gcs = 2 * radius_wire

# Quantities for the calculation of efficiencies
P = 0.5 * ufl.inner(ufl.cross(Esh_3d, ufl.conj(Hsh_3d)), n_3d)
Q = 0.5 * np.imag(eps_au) * k0 * (ufl.inner(E_3d, E_3d)) / Z0 / n_bkg

# Define integration domain for the wire
dAu = dx(au_tag)

# Normalized absorption efficiency
q_abs_fenics_proc = (fem.assemble_scalar(fem.form(Q * dAu)) / gcs / I0).real
# Sum results from all MPI processes
q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

# Normalized scattering efficiency
q_sca_fenics_proc = (fem.assemble_scalar(fem.form(P * dsbc)) / gcs / I0).real

# Sum results from all MPI processes
q_sca_fenics = domain.comm.allreduce(q_sca_fenics_proc, op=MPI.SUM)
```

We compared these efficiencies with analytical ones (defined by a function in `analytical_efficiencies_wire.py`) to test our demo. In the end, we get
an error much smaller than $1\%$, as certified by the final output,and
therefore we can say that our demo works correctly:

```
The analytical absorption efficiency is 1.2115253567863489
The numerical absorption efficiency is 1.210977254477182
The error is 0.04524067994918296%

The analytical scattering efficiency is 0.9481819974744393
The numerical scattering efficiency is 0.947864860367565
The error is 0.033446860172311944%

The analytical extinction efficiency is 2.1597073542607883
The numerical extinction efficiency is 2.158842114844747
The error is 0.040062808247346045%
```

Here below I show the animation of the DOLFINx solution post-processed in
paraview:

![afeaf](images/animation_sbc.gif "Electric field, x-component, real part")

### Demo #2: Perfectly Matched Layers

### Demo #3: Half-loaded waveguide with SLEPc

### Demo #4: Maxwell's equations for axisymmetric geometries

### Bonus tutorial:Hhow to animate solutions in Paraview

## Challenges and final remarks

- what to improve
- what I would make differently....
- ringraziamenti

## To put

Suggestion from Google:

    "Create a blog post or web page or public GitHub gist that describes the work you've done and links to the commits you've made and repositories you've worked on. If there is still work to be done on the project, include that too. You can also share highlights or challenging pieces.

    ❗ This is the best option because it allows you to easily include a lot of information. This is good because it will clearly show what work you did, as well as make it easy for others to use and understand your code."

- Improve ipynb of demos: add title to mesh files, add visualization for waveguide demo and axis demo, add gifs, check why in Docker PyVista does not work.

## Contacts

- Michele Castriotta
  - <i class="fab fa-github"></i> [mikics](https://github.com/mikics)
  - <i class="fab fa-twitter"></i> [@castrimik](https://twitter.com/castrimik)
  - <i class="fab fa-discourse"></i> [CastriMik](https://fenicsproject.discourse.group/u/CastriMik)
  - <i class="fab fa-linkedin"></i> [Michele Castriotta](https://www.linkedin.com/in/michele-castriotta-18aa91a5)
  - <i class="fa fa-envelope"></i> [mich.castriotta@gmail.com](mailto:mich.castriotta@gmail.com)

## Contents

```{tableofcontents}
```
