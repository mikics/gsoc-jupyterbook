# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (DOLFINx complex)
#     language: python
#     name: python3-complex
# ---

# + [markdown] slideshow={"slide_type": "slide"} tags=[] jp-MarkdownHeadingCollapsed=true
# # The Helmholtz equation
#
#
# In this tutorial, we will learn:
#  - How to discretize PDEs with complex-valued fields,
#  - How to import and use high-order meshes from GMSH,
#  - How to use high order discretizations,
#  - How to use UFL expressions.
# -
# ## Problem statement
# We will solve the Helmholtz equation subject to a first order absorbing boundary condition:
#
# $$
# \begin{align*}
# \Delta u + k^2 u &= 0 \qquad \text{in } \Omega,\\
# \nabla u \cdot \mathbf{n} - jku &= g \qquad \text{on } \partial\Omega.
# \end{align*}
# $$ 
# where $k$ is a pieciwise constant wavenumber, and $g$ is the boundary source term computed as:
#
# $$g = \nabla u_i \cdot \mathbf{n} - jku_i$$
#
# and $u_i$ is the incoming plane wave. 

# + slideshow={"slide_type": "slide"} tags=[]
from mpi4py import MPI
from dolfinx import fem
import ufl

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We define some model parameters:

# + slideshow={"slide_type": "fragment"} tags=[]
import numpy as np

# MPI communicator
comm = MPI.COMM_WORLD

# wavenumber in free space (air)
k0 = 10 * np.pi
# Corresponding wavelength
lmbda = 2 * np.pi / k0

# Polynomial degree
degree = 4
# Mesh order
mesh_order = 2

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# This code is only meant to be executed with complex-valued degrees of freedom. To be able to solve such problems, we use the complex build of PETSc.

# + slideshow={"slide_type": "skip"} tags=[]
import sys
from petsc4py import PETSc
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("This tutorial requires complex number support")
    sys.exit(0)
else:
    print(f"Using {PETSc.ScalarType}.")

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We will use [GMSH](http://gmsh.info/) to generate the computational domain (mesh) for this example. As long as GMSH has been installed (including its Python API), DOLFINx supports direct input of GMSH models (generated on one process). DOLFINx will then in turn distribute the mesh over all processes in the communicator passed to `dolfinx.io.gmshio.model_to_mesh`.

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Interfacing with GMSH

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# The function `generate_mesh` creates a GMSH model on rank 0 of `MPI.COMM_WORLD`.

# + slideshow={"slide_type": "skip"} tags=[]
try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

# + tags=[]
import gmsh
from dolfinx.io import gmshio
from mesh_generation import generate_mesh

gmsh.initialize()
model = generate_mesh(lmbda, order=mesh_order)
mesh, cell_tags, _ = gmshio.model_to_mesh(model, comm, 0, gdim=2)
gmsh.finalize()

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Material parameters and boundary conditions

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In this problem, the wave number in the different parts of the domain depends on cell markers, inputted through `cell_tags`.
# We use the fact that a Discontinuous Galerkin space of order 0 (cell-wise piecewise constants) has a one-to-one mapping with the cells local to the process.
# -

DG = fem.FunctionSpace(mesh, ("DG", 0))
k = fem.Function(DG)
k.x.array[:] = k0
k.x.array[cell_tags.find(1)] = 2*k0

# Now we can visualize the wavenumber distribution throughout the computational domain:

# +
import pyvista
import matplotlib.pyplot as plt
from dolfinx.plot import create_vtk_mesh

pyvista.set_jupyter_backend("pythreejs")

topology, cells, geometry = create_vtk_mesh(mesh)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.cell_data["Marker"] = k.x.array.real
plotter = pyvista.Plotter()
renderer = plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we define the boundary source term, by using `ufl.SpatialCoordinate`. When using this function, all quantities using this expression will be evaluated at quadrature points.
# -

n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)
ui = ufl.exp(-1j * k * x[0])
g = ufl.dot(ufl.grad(ui), n) + 1j * k * ui

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Variational form

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we can define the variational problem, using a 4th order Lagrange space. Note that as we are using complex valued functions, we have to use the appropriate inner product, see [DOLFINx tutorial: Complex numbers](https://jorgensd.github.io/dolfinx-tutorial/chapter1/complex_mode.html) for more information.
#
#
# $$ -\int_\Omega \nabla u \cdot \nabla \bar{v} ~ dx + \int_\Omega k^2 u \,\bar{v}~ dx - j\int_{\partial \Omega} ku  \bar{v} ~ ds = \int_{\partial \Omega} g \, \bar{v}~ ds \qquad \forall v \in \widehat{V}. $$

# +
element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
V = fem.FunctionSpace(mesh, element)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


a = -ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    + k**2 * ufl.inner(u, v) * ufl.dx \
    - 1j * k * ufl.inner(u, v) * ufl.ds
L = ufl.inner(g, v) * ufl.ds

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Linear solver

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we will solve the problem using a direct solver (LU).

# + tags=[]
opt = {"ksp_type": "preonly", "pc_type": "lu"}
problem = fem.petsc.LinearProblem(a, L, petsc_options=opt)
uh = problem.solve()
uh.name = "u"

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Visualizing the complex solution
# -
topology, cells, geometry = create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.point_data["Re(u)"] = uh.x.array.real
grid.point_data["Im(u)"] = uh.x.array.imag


# + slideshow={"slide_type": "skip"} tags=[]
pyvista.start_xvfb(0.5) # Start virtual framebuffer for plotting

import matplotlib.pyplot as plt
def plot_function(grid, name):
    plotter = pyvista.Plotter()
    grid.set_active_scalars(name)
    renderer = plotter.add_mesh(grid, show_edges=False)
    img = plotter.screenshot(f"{name}.png",
                         transparent_background=True,
                         window_size=(1000,1000))
    plt.axis("off")
    plt.gcf().set_size_inches(10,10)
    fig = plt.imshow(img)



# + slideshow={"slide_type": "fragment"} tags=[]
plot_function(grid, "Re(u)")

# + slideshow={"slide_type": "notes"} tags=[]
plot_function(grid, "Im(u)")

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Saving higher order functions

# + tags=[]
from dolfinx.io import XDMFFile, VTXWriter

# XDMF write the solution as a P1 function
with XDMFFile(comm, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)

# VTX can write higher order function
with VTXWriter(comm, "out_high_order.bp", [uh]) as f:
    f.write(0.0)
