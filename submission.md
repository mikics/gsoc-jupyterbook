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

We compared these efficiencies with analytical ones (calculated by a function in `analytical_efficiencies_wire.py`) to test our demo. In the end, we get
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

As a final treat, here below you can see the animation of the DOLFINx solution post-processed in
paraview:

![afeaf](images/animation_sbc.gif "Electric field, x-component, real part")

### Demo #2: Perfectly Matched Layers

In the second demo, we show how to implement perfectly matched layers (shortly PMLs) for
the same problem solved in the first demo, i.e. the
scattering of a TM-polarized plane wave by an infinite gold wire. Perfectly
matched layers are artificial layers that gradually absorb outgoing waves impinging on them, and are extensively used in time-harmonic electromagnetic
problems for domain truncation. However, their mathematical implementation can
be quite tricky, and therefore showing the math and the corresponding DOLFINx
implementation is crucial for allowing users to quickly apply the FEniCSx environment to their electromagnetic problems.

For this demo, we chose to use a square PML layer. In order to define PML, we
implement in the PML domain a complex coordinate transformation of this kind:

$$
\begin{align}
& x^\prime= x\left\{1+j\frac{\alpha}{k_0}\left[\frac{|x|-l_{dom}/2}
{(l_{pml}/2 - l_{dom}/2)^2}\right] \right\}\\

& y^\prime= y\left\{1+j\frac{\alpha}{k_0}\left[\frac{|y|-l_{dom}/2}
{(l_{pml}/2 - l_{dom}/2)^2}\right] \right\}\\
\end{align}
$$

We then calculate the Jacobian associated with this transformation:

$$
\mathbf{J}=\mathbf{A}^{-1}= \nabla\boldsymbol{x}^
\prime(\boldsymbol{x}) =
\left[\begin{array}{ccc}
\frac{\partial x^{\prime}}{\partial x} &
\frac{\partial y^{\prime}}{\partial x} &
\frac{\partial z^{\prime}}{\partial x} \\
\frac{\partial x^{\prime}}{\partial y} &
\frac{\partial y^{\prime}}{\partial y} &
\frac{\partial z^{\prime}}{\partial y} \\
\frac{\partial x^{\prime}}{\partial z} &
\frac{\partial y^{\prime}}{\partial z} &
\frac{\partial z^{\prime}}{\partial z}
\end{array}\right]=\left[\begin{array}{ccc}
\frac{\partial x^{\prime}}{\partial x} & 0 & 0 \\
0 & \frac{\partial y^{\prime}}{\partial y} & 0 \\
0 & 0 & \frac{\partial z^{\prime}}{\partial z}
\end{array}\right]=\left[\begin{array}{ccc}
J_{11} & 0 & 0 \\
0 & J_{22} & 0 \\
0 & 0 & 1
\end{array}\right]
$$

Finally, we can express the complex coordinate transformation as a material
transformation within the PML, therefore having the following anisotropic,
inhomogeneous, and complex relative permittivity $\boldsymbol{\varepsilon}_{pml}$ and permeability
$\boldsymbol{\mu}_{pml}$:

$$
\begin{align}
& {\boldsymbol{\varepsilon}_{pml}} =
\det(\mathbf{A}^{-1}) \mathbf{A} {\boldsymbol{\varepsilon}_b}\mathbf{A}^{T},\\
& {\boldsymbol{\mu}_{pml}} =
\det(\mathbf{A}^{-1}) \mathbf{A} {\boldsymbol{\mu}_b}\mathbf{A}^{T},
\end{align}
$$

All these steps have been defined in DOLFINx using the following functions:

```
def pml_coordinates(x, alpha: float, k0: complex,
                    l_dom: float, l_pml: float):
    return (x + 1j * alpha / k0 * x
            * (algebra.Abs(x) - l_dom / 2)
            / (l_pml / 2 - l_dom / 2)**2

def create_eps_mu(pml, eps_bkg, mu_bkg):

    J = grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = as_matrix(((J[0, 0], 0, 0),
                   (0, J[1, 1], 0),
                   (0, 0, 1)))

    A = inv(J)
    eps_pml = det(J) * A * eps_bkg * transpose(A)
    mu_pml = det(J) * A * mu_bkg * transpose(A)
    return eps_pml, mu_pml

# PML corners
xy_pml = as_vector((pml_coordinates(x[0], alpha, k0, l_dom, l_pml),
                    pml_coordinates(x[1], alpha, k0, l_dom, l_pml)))

# PML rectangles along x
x_pml = as_vector((pml_coordinates(x[0], alpha, k0, l_dom, l_pml), x[1]))

# PML rectangles along y
y_pml = as_vector((x[0], pml_coordinates(x[1], alpha, k0, l_dom, l_pml)))

eps_x, mu_x = create_eps_mu(x_pml, eps_bkg, 1)
eps_y, mu_y = create_eps_mu(y_pml, eps_bkg, 1)
eps_xy, mu_xy = create_eps_mu(xy_pml, eps_bkg, 1)
```

We need to define multiple coordinate transformation since in the corners of the PML both coordinates are transformed, while in the rest of the PML just
one of them is. In the end, we can implement the weak form in DOLFINx in this way:

$$
\begin{align}
&\int_{\Omega_{pml}}\left[\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}
\right]\cdot \nabla \times \bar{\mathbf{v}}-k_{0}^{2}
\left[\boldsymbol{\varepsilon}_{pml} \mathbf{E} \right]\cdot
\bar{\mathbf{v}}~ d x  \\
+ &\int_{\Omega_m\cup\Omega_b}-(\nabla \times \mathbf{E}_s)
\cdot (\nabla \times \bar{\mathbf{v}})+\varepsilon_{r} k_{0}^{2}
\mathbf{E}_s \cdot \bar{\mathbf{v}}+k_{0}^{2}\left(\varepsilon_{r}
-\varepsilon_b\right)\mathbf{E}_b \cdot \bar{\mathbf{v}}~\mathrm{d}x.
= 0.
\end{align}
$$

```
F = - inner(curl_2d(Es), curl_2d(v)) * dDom \
    + eps * k0 ** 2 * inner(Es, v) * dDom \
    + k0 ** 2 * (eps - eps_bkg) * inner(Eb, v) * dDom \
    - inner(inv(mu_x) * curl_2d(Es), curl_2d(v)) * dPml_x \
    - inner(inv(mu_y) * curl_2d(Es), curl_2d(v)) * dPml_y \
    - inner(inv(mu_xy) * curl_2d(Es), curl_2d(v)) * dPml_xy \
    + k0 ** 2 * inner(eps_x * Es_3d, v_3d) * dPml_x \
    + k0 ** 2 * inner(eps_y * Es_3d, v_3d) * dPml_y \
    + k0 ** 2 * inner(eps_xy * Es_3d, v_3d) * dPml_x
```

Then, as in the first demo, we calculated the efficiencies and compared them with analytical ones. For scattering efficiencies, we needed to define
the integration line slightly differently with respect to demo #1, since
we have to deal with an inner facet:

```
marker = fem.Function(D)
scatt_facets = facet_tags.find(scatt_tag)
incident_cells = mesh.compute_incident_entities(domain, scatt_facets,
                                                domain.topology.dim - 1,
                                                domain.topology.dim)

midpoints = mesh.compute_midpoints(domain, domain.topology.dim, incident_cells)
inner_cells = incident_cells[(midpoints[:, 0]**2
                              + midpoints[:, 1]**2) < (l_scatt)**2]

marker.x.array[inner_cells] = 1

# Quantities for the calculation of efficiencies
P = 0.5 * inner(cross(Esh_3d, conj(Hsh_3d)), n_3d) * marker

# Define integration facet for the scattering efficiency
dS = Measure("dS", domain, subdomain_data=facet_tags)

# Normalized scattering efficiency
q_sca_fenics_proc = (fem.assemble_scalar(
    fem.form((P('+') + P('-')) * dS(scatt_tag))) / gcs / I0).real

```

As in the first demo, even in this case the error is under $1\%$:

```
The analytical absorption efficiency is 0.9089500187622276
The numerical absorption efficiency is 0.9075812357239408
The error is 0.1505894724718481%

The analytical scattering efficiency is 0.8018061316558375
The numerical scattering efficiency is 0.7996621815340356
The error is 0.2673900880970269%

The analytical extinction efficiency is 1.710756150418065
The numerical extinction efficiency is 1.7072434172579762
The error is 0.2053321953120203%
```

Here below, you can see the time-harmonic animation of the scattered
electric field norm obtained in DOLFINx and post-processed in Paraview:

![afeaf](images/animation_pml.gif "Norm of Electric field")

### Demo #3: Half-loaded waveguide with SLEPc

The third demo shows how to solve an eigenvalue electromagnetic problem
in DOLFINx with the SLEPc library. In particular, we solve the eigenvalue problem of a half-loaded electromagnetic waveguide with perfect
electric conducting walls. The equations for our problem are:

$$
\begin{align}
&\nabla \times \frac{1}{\mu_{r}} \nabla \times \mathbf{E}-k_{0}^{2}
\epsilon_{r} \mathbf{E}=0 \quad &\text { in } \Omega\\
&\hat{n}\times\mathbf{E} = 0 &\text { on } \Gamma
\end{align}
$$

The final weak form can be found by considering a known $z$ dependance of
the electric field:

$$
\mathbf{E}(x, y, z)=\left[\mathbf{E}_{t}(x, y)+\hat{z} E_{z}(x, y)\right]
e^{-jk_z z}
$$

and by using the following substitution:

$$
\begin{align}
& \mathbf{e}_t = k_z\mathbf{E}_t\\
& e_z = -jE_z
\end{align}
$$

In the end, we get the following equation:
$$
\begin{aligned}
F_{k_z}(\mathbf{e})=\int_{\Omega} &\left(\nabla_{t} \times
\mathbf{e}_{t}\right) \cdot\left(\nabla_{t} \times
\bar{\mathbf{v}}_{t}\right) -k_{o}^{2} \epsilon_{r} \mathbf{e}_{t} \cdot
\bar{\mathbf{v}}_{t} \\
&+k_z^{2}\left[\left(\nabla_{t} e_{z}+\mathbf{e}_{t}\right)
\cdot\left(\nabla_{t} \bar{v}_{z}+\bar{\mathbf{v}}_{t}\right)-k_{o}^{2}
\epsilon_{r} e_{z} \bar{v}_{z}\right] \mathrm{d} x = 0
\end{aligned}
$$

which we can write in a more compact form as:

$$
\left[\begin{array}{cc}
A_{t t} & 0 \\
0 & 0
\end{array}\right]\left\{\begin{array}{l}
\mathbf{e}_{t} \\
e_{z}
\end{array}\right\}=-k_z^{2}\left[\begin{array}{ll}
B_{t t} & B_{t z} \\
B_{z t} & B_{z z}
\end{array}\right]\left\{\begin{array}{l}
\mathbf{e}_{t} \\
e_{z}
\end{array}\right\}
$$

This problem is a *generalized eigenvalue problem*, where the eigenvalues are all the possible $-k_z^2$ and $\mathbf{e}_t, e_z$ substained by the structure.

The weak form in DOLFINx can be written in this way:

```
a_tt = (inner(curl(et), curl(vt)) - k0
        ** 2 * eps * inner(et, vt)) * dx
b_tt = inner(et, vt) * dx
b_tz = inner(et, grad(vz)) * dx
b_zt = inner(grad(ez), vt) * dx
b_zz = (inner(grad(ez), grad(vz)) - k0
        ** 2 * eps * inner(ez, vz)) * dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)
```

While for the perfect electric conductor condition we used these commands:

```
bc_facets = exterior_facet_indices(domain.topology)

bc_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, bc_facets)

u_bc = fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)
```

Then, we assembled the $A$ and $B$ matrices with PETSc:

```
A = fem.petsc.assemble_matrix(a, bcs=[bc])
A.assemble()
B = fem.petsc.assemble_matrix(b, bcs=[bc])
B.assemble()
```

These matrices are the inputs SLEPc needs to solve our problem, as shown in the next snippet:

```
eps = SLEPc.EPS().create(domain.comm)

eps.setOperators(A, B)
```

Then, we need to tweak some SLEPc settings, to guarantee the convergence of the solver, as shown below. For this problem, we need to use a spectral transformation to solve the problem, and getting our eigenvalue(s). In particular, spectral transformation techniques map the eigenvalues in other portion of the spectrum, to make the algorithm more efficient. We then need to set a target value for our eigenvalue, the number of eigenvalues we want to find, and we can finally solve the problem. It is worth highlighting that solving eigenvalue problems can be tricky, and therefore finding the correct eigenvalues of the problem may require a lot of tweaking.

```
# Set the tolerance for the solution
eps.setTolerances(tol=tol)

# Set solver type
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

# Set spectral transformation
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)

# Set type of target
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

# Set target
eps.setTarget(-(0.5 * k0)**2)

# Set number of eigenvalues
eps.setDimensions(nev=1)

# Solve
eps.solve()
```

Then, we verify if the solutions from SLEPc are consistent with the analytical formula for the half-loaded waveguide modes, which are:

$$
\begin{aligned}
\textrm{For TE}_x \textrm{ modes}:
\begin{cases}
&k_{x d}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=k_0^{2}
\varepsilon_{d} \\
&k_{x v}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=k_0^{2}
\varepsilon_{v} \\
& k_{x d} \cot k_{x d} d + k_{x v} \cot \left[k_{x v}(h-d)\right] = 0
\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
\textrm{For TM}_x \textrm{ modes}:
\begin{cases}
&k_{x d}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=
k_0^{2} \varepsilon_{d} \\
&k_{x v}^{2}+\left(\frac{n \pi}{b}\right)^{2}+k_{z}^{2}=
k_0^{2} \varepsilon_{v} \\
& \frac{k_{x d}}{\varepsilon_{d}} \tan k_{x d} d +
\frac{k_{x v}}{\varepsilon_{v}} \tan \left[k_{x v}(h-d)\right] = 0
\end{cases}
\end{aligned}
$$

The `analytical_modes.py` file defines the functions for doing the verification.

In the end, we get the following eigenvalue, and the corresponding $k_z$:

```
eigenvalue: (-1.6924040028250327+1.3702668664033287e-14j)
kz: (1.3009242878911258-5.266512736973384e-15j)
kz/k0: (0.4658591947638973-1.885930953627917e-15j)
```

The eigenvalue successfuly passes the verification step, and therefore it satisfies the mode equations (up to a certain threshold).

### Demo #4: Maxwell's equations for axisymmetric geometries

### Bonus tutorial: How to animate solutions in Paraview

## Challenges and final remarks

- what to improve
- what I would make differently....
- ringraziamenti

## To do

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
