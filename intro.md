# GSoC 2022 - Expanding FEniCSx electromagnetic demos

## Summary of my GSoC experience

### List of contributions

#### Pull Requests

- [PR #2237](https://github.com/FEniCS/dolfinx/pull/2237) (open): this pull request
adds the demo showing how to set a time-harmonic electromagnetic problem
with scattering boundary conditions:
  - `demo_scattering_boundary_conditions.py`:
  - `mesh_wire.py`:
  - `analytical_efficiencies_wire.py`:
- [PR #2276](https://github.com/FEniCS/dolfinx/pull/2276) (open): this pull request
adds the demo showing how to set a time-harmonic electromagnetic problem with perfectly matched layers:
  - `demo_pml.py`:
  - `mesh_wire_pml.py`:
  - `analytical_efficiencies_wire.py`:
- [PR #2338](https://github.com/FEniCS/dolfinx/pull/2338) (open): this pull request adds
the demo showing how to solve an electromagnetic eigenvalue problem problem with DOLFINx and SLEPc:
  - `demo_waveguide.py`:
  - `analytical_efficiencies_wire.py`:
- [PR #2339](https://github.com/FEniCS/dolfinx/pull/2339) (open): this pull request adds the demo
showing how to solve an time-harmonic electromagnetic problem for axisymmetric geometry:
  - `demo_scattering_boundary_conditions.py`:
  - `mesh_wire.py`:
  - `analytical_efficiencies_wire.py`:
- [PR #2357](https://github.com/FEniCS/dolfinx/pull/2357) (merged): this pull request adds the line
`from dolfinx.io import gmshio` in `python/io/__init__.py` so that `gmshio` is considered a module
in DOLFINx. Besides, it also removes an unnecessary `gmshio` string in the `has_adios2` conditional
block.

#### Issues

- [GH issue #2343](https://github.com/FEniCS/dolfinx/issues/2343) (closed as completed):

#### Others

#### What's next

### To put

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
