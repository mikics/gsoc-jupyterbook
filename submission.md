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

![sbc](images/animation_sbc.gif "Electric field, x-component, real part")

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

![pml](images/animation_pml.gif "Norm of Electric field")

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

The fourth demo shows how to solve Maxwell's equations for a simple three-dimensional axisymmetric geometry, i.e. a sphere. Generally, solving three-dimensional electromagnetic problems is computationally expensive, and it may result in prohibitive memory and time requirements. However, if the three-dimensional geometry has an axisymmetry, the full wave problem can be decomposed in few two-dimensional problems, with an overall much lower computational cost.

We start from the weak form for Maxwell's equations and PML equations:

$$
\begin{align}
&\int_{\Omega_m\cup\Omega_b}-(\nabla \times \mathbf{E}_s)
\cdot (\nabla \times \bar{\mathbf{v}})+\varepsilon_{r} k_{0}^{2}
\mathbf{E}*s \cdot \bar{\mathbf{v}}+k*{0}^{2}\left(\varepsilon_{r}
-\varepsilon_b\right)\mathbf{E}_b \cdot \bar{\mathbf{v}}~\mathrm{d} x\\
+&\int_{\Omega_{pml}}\left[\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}_s
\right]\cdot \nabla \times \bar{\mathbf{v}}-k_{0}^{2}
\left[\boldsymbol{\varepsilon}_{pml} \mathbf{E}_s \right]\cdot
\bar{\mathbf{v}}~ d x=0
\end{align}
$$

We then decompose the fields in cylindrical harmonics:

$$
\begin{align}
& \mathbf{E}_s(\rho, z, \phi) = \sum_m\mathbf{E}^{(m)}_s(\rho, z)e^{-im\phi} \\
& \mathbf{E}_b(\rho, z, \phi) = \sum_m\mathbf{E}^{(m)}_b(\rho, z)e^{-im\phi} \\
& \bar{\mathbf{v}}(\rho, z, \phi) =
\sum_m\bar{\mathbf{v}}^{(m)}(\rho, z)e^{+im\phi}\\
\end{align}
$$

and with few other steps (that we are going to skip here but are extensively explained in the demo) we arrive at the final weak form, showing 1) that the problem is formulated over a two-dimensional domain, and 2) that the different cylindrical harmonics propagate independently:

$$
\begin{align}
\sum_{m}\int_{\Omega_{cs}}&-(\nabla \times \mathbf{E}^{(m)}_s)
\cdot (\nabla \times \bar{\mathbf{v}}^{(m)})+\varepsilon_{r} k_{0}^{2}
\mathbf{E}^{(m)}_s \cdot \bar{\mathbf{v}}^{(m)}
+k_{0}^{2}\left(\varepsilon_{r}
-\varepsilon_b\right)\mathbf{E}^{(m)}_b \cdot \bar{\mathbf{v}}^{(m)}\\
&+\left(\boldsymbol{\mu}^{-1}_{pml} \nabla \times \mathbf{E}^{(m)}_s
\right)\cdot \nabla \times \bar{\mathbf{v}}^{(m)}-k_{0}^{2}
\left(\boldsymbol{\varepsilon}_{pml} \mathbf{E}^{(m)}_s \right)\cdot
\bar{\mathbf{v}}^{(m)}~ \rho d\rho dz =0
\end{align}
$$

Therefore, the original problem can be solved for each cylindrical harmonic over a 2D cross-section of the original domain. For the sake of simplicity, we choose this cross-section to be the one at $\phi = 0$.

In the demo we present and implement a lof of concepts that we need for axisymmetric problems. We list them in the following sections.

#### $\nabla\times$ operator for cylindrical coordinates

In cylindrical coordinates, the curl operator becomes:

$$
\begin{align}
\left(\nabla \times \mathbf{a}^{(m)}\right) = &\left[\hat{\rho}
\left(-\frac{\partial a_{\phi}^{(m)}}{\partial z}
-i \frac{m}{\rho} a_{z}^{(m)}\right)+\\ \hat{\phi}
\left(\frac{\partial a_{\rho}^{(m)}}{\partial z}
-\frac{\partial a_{z}^{(m)}}{\partial \rho}\right)+\right.\\
&\left.+\hat{z}\left(\frac{a_{\phi}^{(m)}}{\rho}
+\frac{\partial a_{\phi}^{(m)}}{\partial \rho}
+i \frac{m}{\rho} a_{\rho}^{(m)}\right)\right]
\end{align}
$$

The corresponding DOLFINx implementation is:

```
def curl_axis(a, m: int, rho):

    curl_r = -a[2].dx(1) - 1j * m / rho * a[1]
    curl_z = a[2] / rho + a[2].dx(0) + 1j * m / rho * a[0]
    curl_p = a[0].dx(1) - a[1].dx(0)

    return ufl.as_vector((curl_r, curl_z, curl_p))
```

#### Cylindrical harmonic expansion for $\mathbf{E}^{(m)}_b$

The $m$-th harmonic for the background field can be expressed in terms of Bessel functions as:

$$
\begin{split}
\begin{align}
\mathbf{E}^{(m)}_b = &\hat{\rho} \left(E_{0} \cos \theta
e^{i k z \cos \theta} i^{-m+1} J_{m}^{\prime}\left(k_{0} \rho \sin
\theta\right)\right)\\
+&\hat{z} \left(E_{0} \sin \theta e^{i k z \cos \theta}i^{-m} J_{m}
\left(k \rho \sin \theta\right)\right)\\
+&\hat{\phi} \left(\frac{E_{0} \cos \theta}{k \rho \sin \theta}
e^{i k z \cos \theta} i^{-m} J_{m}\left(k \rho \sin \theta\right)\right)
\end{align}
\end{split}
$$

In DOLFINx, the corresponding implementation is:

```
def background_field_rz(theta: float, n_bkg: float, k0: float, m: int, x):

    k = k0 * n_bkg

    a_r = (np.cos(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**(-m + 1) * jvp(m, k * x[0] * np.sin(theta), 1))

    a_z = (np.sin(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**-m * jv(m, k * x[0] * np.sin(theta)))

    return (a_r, a_z)

def background_field_p(theta: float, n_bkg: float, k0: float, m: int, x):

    k = k0 * n_bkg

    a_p = (np.cos(theta) / (k * x[0] * np.sin(theta))
           * np.exp(1j * k * x[1] * np.cos(theta)) * m
           * (1j)**(-m) * jv(m, k * x[0] * np.sin(theta)))

    return a_p
```

#### Axisymmetric PMLs

For axisymmetric structure, we need an axisymmetric complex coordinate transformation
for PML. One possible choice is:

$$
\begin{split}
\begin{align}
& \rho^{\prime} = \rho\left[1 +j \alpha/k_0 \left(\frac{r

- r_{dom}}{r~r_{pml}}\right)\right] \\
& z^{\prime} = z\left[1 +j \alpha/k_0 \left(\frac{r
- r_{dom}}{r~r_{pml}}\right)\right] \\
& \phi^{\prime} = \phi \\
\end{align}
\end{split}
$$

and the corresponding Jacobian is:

$$
\begin{split}
\mathbf{J}=\mathbf{A}^{-1}= \nabla\boldsymbol{\rho}^
\prime(\boldsymbol{\rho}) =
\left[\begin{array}{ccc}
\frac{\partial \rho^{\prime}}{\partial \rho} &
\frac{\partial z^{\prime}}{\partial \rho} &
0 \\
\frac{\partial \rho^{\prime}}{\partial z} &
\frac{\partial z^{\prime}}{\partial z} &
0 \\
0 &
0 &
\frac{\rho^\prime}{\rho}\frac{\partial \phi^{\prime}}{\partial \phi}
\end{array}\right]=\left[\begin{array}{ccc}
J_{11} & J_{12} & 0 \\
J_{21} & J_{22} & 0 \\
0 & 0 & J_{33}
\end{array}\right]
\end{split}
$$

In DOLFINx, similarly to what we did for demo #2, we can define these functions
as:

```
def pml_coordinate(
        x, r, alpha: float, k0: float, radius_dom: float, radius_pml: float):

    return (x + 1j * alpha / k0 * x * (r - radius_dom) / (radius_pml * r))


def create_eps_mu(pml, rho, eps_bkg, mu_bkg):

    J = ufl.grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = ufl.as_matrix(((J[0, 0], J[0, 1], 0),
                       (J[1, 0], J[1, 1], 0),
                       (0, 0, pml[0] / rho)))

    A = ufl.inv(J)
    eps_pml = ufl.det(J) * A * eps_bkg * ufl.transpose(A)
    mu_pml = ufl.det(J) * A * mu_bkg * ufl.transpose(A)
    return eps_pml, mu_pml

rho, z = ufl.SpatialCoordinate(domain)
alpha = 5
r = ufl.sqrt(rho**2 + z**2)

pml_coords = ufl.as_vector((
    pml_coordinate(rho, r, alpha, k0, radius_dom, radius_pml),
    pml_coordinate(z, r, alpha, k0, radius_dom, radius_pml)))

eps_pml, mu_pml = create_eps_mu(pml_coords, rho, eps_bkg, 1)

```

#### Solving the problem

The problem needs to be solved over many $m\in \Z$. Thanks to Bessel functions parities,
we can only solve for $m\geq0$, and adding a $2$ factor to solutions for $m\geq1$.
Another question is: where do we stop the expansion? For deeply sub-wavelength particles, as in our case, few cylindrical harmonics ($m = 0, 1$) are usually enough to reach a good accuracy.

Therefore, the problem can be solved in DOLFINx with a loop over the $m$:

```
m_list = [0, 1]

for m in m_list:

    # Definition of Trial and Test functions
    Es_m = ufl.TrialFunction(V)
    v_m = ufl.TestFunction(V)

    # Background field
    Eb_m = fem.Function(V)
    f_rz = partial(background_field_rz, theta, n_bkg, k0, m)
    f_p = partial(background_field_p, theta, n_bkg, k0, m)
    Eb_m.sub(0).interpolate(f_rz)
    Eb_m.sub(1).interpolate(f_p)

    curl_Es_m = curl_axis(Es_m, m, rho)
    curl_v_m = curl_axis(v_m, m, rho)

    F = - ufl.inner(curl_Es_m, curl_v_m) * rho * dDom \
        + eps * k0 ** 2 * ufl.inner(Es_m, v_m) * rho * dDom \
        + k0 ** 2 * (eps - eps_bkg) * ufl.inner(Eb_m, v_m) * rho * dDom \
        - ufl.inner(ufl.inv(mu_pml) * curl_Es_m, curl_v_m) * rho * dPml \
        + k0 ** 2 * ufl.inner(eps_pml * Es_m, v_m) * rho * dPml

    a, L = ufl.lhs(F), ufl.rhs(F)

    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                      "ksp_type": "preonly", "pc_type": "lu"})
    Esh_m = problem.solve()

    if m == 0:

        Esh.x.array[:] = Esh_m.x.array[:] * np.exp(- 1j * m * phi)

    elif m == m_list[0]:

        Esh.x.array[:] = 2 * Esh_m.x.array[:] * np.exp(- 1j * m * phi)

    else:

        Esh.x.array[:] += 2 * Esh_m.x.array[:] * np.exp(- 1j * m * phi)
```

#### Test

The DOLFINx solution was tested by calculating the numerical efficiencies and
comparing them with the analytical efficiencies. The analytical efficiencies were
calculated with the [scattnlay](https://github.com/ovidiopr/scattnlay) library,
with the following call:

```
from scattnlay import scattnlay

m = np.sqrt(eps_au)/n_bkg
x = 2*np.pi*radius_sph/wl0*n_bkg

q_ext, q_sca, q_abs = scattnlay(np.array([x], dtype=np.complex128), np.array([m], dtype=np.complex128))[1:4]
```

The calculation of the numerical efficiencies is similar to what we did in
demo #2, with the only difference that we need to add a $2$ factor for the
efficiencies resulting from $m\geq1$ harmonics.

The final error for all efficiencies is well below $1\%$, as shown by the final output in the demo, and therefore we can safely say that the problem in DOLFINx is well
set:

```
The analytical absorption efficiency is 0.9622728008329892
The numerical absorption efficiency is 0.9583126377885698
The error is 0.41154265619804153%

The analytical scattering efficiency is 0.07770397394691526
The numerical scattering efficiency is 0.07737655859424392
The error is 0.4213624297967401%

The analytical extinction efficiency is 1.0399767747799045
The numerical extinction efficiency is 1.0356891963828136
The error is 0.4122763604983602%
```

As a final example, here below we show the magnitude of the real part of the (total)
scattered field:

![axis](images/axis.png "Magnitude of the real part of the scattered electric field")

## Challenges and final remarks

The most challenging aspect of GSoC was dealing with unexpected results and
troubleshooting. Indeed, debugging was one of the most important skill I needed to improve during GSoC, since it never happens to run the code at the first try without problems, and therefore one should be able to quickly understand how to approach
and solve these situations.

For all demos I needed a certain degree of debugging, but for demo #4 it was particularly tough. Indeed, I had to deal with an unexpected behaviour of the demo:
whenever I used `degree = 3` (or higher) discretization elements, the error for the efficiencies increased, and DOLFINx solutions had some nasty artifacts, as shown in
the image below.

![error](images/error.png "Artifacts in the field for high degree elements")

In order to understand the error, I tried to isolate the problem and collect as much
information as possible. The first thing that came to my mind was a bug in the
functions I wrote for the demo. Therefore I substituted PML with scattering
boundary conditions, and verified if the background field showed the same artifacts
of the output field. However, scattering boundary conditions did not change the
output (high errors, same nasty artifacts), and the background field had no strange
behavior, and was consistent with the background field I implemented
for the same problem in legacy DOLFIN (extensively tested up to `degree = 5`). Therefore, I was reasonably sure that PML and
the background field were implemented correctly. What I tried to do next was
comparing my legacy DOLFIN demo with my new DOLFINx demo for different
harmonic numbers. What I noticed was that for `m = 0` both demos output the
same efficiencies whatever the degree, while these efficiencies diverged for
`m = 1` solutions. Therefore, something happened when passing from `m = 0` and
`m = 1`. The only two Python objects affected by the harmonic numbers were the
`background_field_rz`, `background_field_p`, and the `curl_axis` functions. As
already said, the background field functions were doing their job correctly, and
it was then clear that something wrong was happening within `curl_axis`, and
in particular for the terms activated by `m = 1`. At this point, the problem
was clearly something wrong in the DOLFINx codebase, and together with my
mentors I decided to open a GitHub issue and to "pass the ball" to more
experienced developers. It turned out that there was a bug in the permuation
inside `MixedElement`, solved by [PR #2347](https://github.com/FEniCS/dolfinx/pull/2347).

When dealing with these issues, it was particularly important to not get frustrated,
and to build a strong communications with my mentors. Therefore, my suggestions for new
GSoC contributors are: 1) do not work on an issue if you are too much tired, rather take
some time off and try again after few hours or the next day, and 2) ask for help if
you are running out of ideas!

In the end, the GSoC has been the best professional experience of my life, since I have
learnt
so much in 12 weeks that I can hardly recall everything. Just to name a few: I have learnt how to run and manage Docker containers, I have a deeper knowledge of the
FEniCSx environment, I can now run and build static websites as the one you are reading now, I have a deeper understanding of how a big project as FEniCSx is managed, and I have
gained much more confidence in using git and GitHub tools.

My only regret is that I have not been too much active on the [FEniCS discourse group](https://fenicsproject.discourse.group/)
for helping other users, mainly because answering many of the posted questions require a deep
technical knowledge about the FEniCSx environment that I have not gained yet. However,
I have also understood that solving other users' problems is a great way to better understand FEniCSx, and therefore I will be for sure more active in the future.

Last but not least, I would like to thank my mentors Jørgen S. Dokken, Igor Baratta and Jack Hale for their patience and help over the 12 weeks. Discussing and working with them has been great since the beginning, and I could not have wished for a better
mentors/mentee relationship. I would also like to thank my supervisor Cristian
Ciracì, which introduced me to FEniCS and which helped me so much when solving
the problems I show in the demo.

And, if you arrived here, I would like to thank you too!

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
