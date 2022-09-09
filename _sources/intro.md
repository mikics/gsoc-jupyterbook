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
the Jupyter notebook of the demos I have built. In this way, you can directly
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

- [PR #2237](https://github.com/FEniCS/dolfinx/pull/2237) (open): this pull request
adds the demo showing how to set a time-harmonic electromagnetic problem
with scattering boundary conditions:
  - `demo_scattering_boundary_conditions.py`: solves the time-harmonic problem
  of a TM-polarized plane wave scattered by a gold wire using
  scattering boundary conditions;
  - `mesh_wire.py`: generates the mesh for the demo;
  - `analytical_efficiencies_wire.py`: calculates the analytical efficiencies
  for a wire;
- [PR #2276](https://github.com/FEniCS/dolfinx/pull/2276) (open): this pull request
adds the demo showing how to set a time-harmonic electromagnetic problem with perfectly matched layers:
  - `demo_pml.py`: demo for solving the time-harmonic problem of a TM-polarized plane wave scattered by a gold wire using perfectly matched layers;
  - `mesh_wire_pml.py`: generates the mesh for the demo;
  - `analytical_efficiencies_wire.py`: calculates the analytical efficiencies
  for a wire;
- [PR #2338](https://github.com/FEniCS/dolfinx/pull/2338) (open): this pull request adds
the demo showing how to solve a time-harmonic electromagnetic eigenvalue problem problem with DOLFINx and SLEPc:
  - `demo_waveguide.py`: solves the eigenvalue problem associated with an electromagnetic half-loaded
  waveguide with SLEPc;
  - `analytical_modes.py`: verifies if FEniCSx modes satisfy the analytical equations for the
  half-loaded waveguide;
- [PR #2339](https://github.com/FEniCS/dolfinx/pull/2339) (open): this pull request adds the demo
showing how to solve a time-harmonic electromagnetic problem for axisymmetric geometry:
  - `demo_axis.py`: solves the time-harmonic problem of a plane wave scattered by a sphere
  within the axisymmetric approximation, using perfectly matched layers
  - `mesh_sphere_axis.py`: generates the mesh for the demo
- [PR #2357](https://github.com/FEniCS/dolfinx/pull/2357) (merged): this pull request adds the line
`from dolfinx.io import gmshio` in `python/io/__init__.py` so that `gmshio` is considered a module
in DOLFINx. Besides, it also removes an unnecessary `gmshio` string in the `has_adios2` conditional
block.

### Issues

- [GH issue #2343](https://github.com/FEniCS/dolfinx/issues/2343) (closed as completed): issue
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

## Challenges and final remarks

## To put

Suggestion from Google:

    "Create a blog post or web page or public GitHub gist that describes the work you've done and links to the commits you've made and repositories you've worked on. If there is still work to be done on the project, include that too. You can also share highlights or challenging pieces.

    ❗ This is the best option because it allows you to easily include a lot of information. This is good because it will clearly show what work you did, as well as make it easy for others to use and understand your code."

- list of PR, with their current status
- list of GH issue, with their current status
- paraview animation
- what's next
- highlights and challenges
- Further remarks, what to improve, what I would do differently....
- Summary of demos, tests, PRs and GH issues
- Improve ipynb of demos: add title to mesh files, add visualization for waveguide demo and axis demo, add gifs, check why in Docker PyVista does not work.
- Use extrude-rotate in pyvista for the axisymmetric plots

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
