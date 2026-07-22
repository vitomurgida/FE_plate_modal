from dolfinx import fem

def plot_mode_shape(eps, V, mode, freq_mode, warp_scale, bc):
    """
    Plot the transverse displacement w of a selected eigenmode and save it.

    Parameters
    ----------
    eps : SLEPc.EPS
        The solved eigenproblem.
    V : dolfinx.fem.FunctionSpace
        Mixed space (w, theta).
    mode : int
        Mode number to plot (1-based: 1,2,3,...).
    freq_mode : float or str
        The mode frequency to display in the title and filename.
    warp_scale : float
        Visual scaling factor for deformation.
    bc : str
        The boundary condition label used in the title and filename.
    """
    import numpy as np
    from petsc4py import PETSc

    # PyVista-based visualization
    import pyvista as pv
    from dolfinx import plot

    i = mode - 1
    nconv = eps.getConverged()
    if i < 0 or i >= nconv:
        raise ValueError(f"Requested mode={mode}, but only {nconv} modes converged.")

    # Create function and fill with eigenvector
    uh = fem.Function(V)
    vr, _ = V.dofmap.index_map.size_local, V.dofmap.index_map.num_ghosts

    # SLEPc gives eigenvector in PETSc Vec
    xr = PETSc.Vec().createMPI(V.dofmap.index_map.size_global, comm=V.mesh.comm)
    xi = PETSc.Vec().createMPI(V.dofmap.index_map.size_global, comm=V.mesh.comm)
    eps.getEigenvector(i, xr, xi)

    # Copy local part into Function
    uh.x.array[:] = xr.getArray(readonly=True)[: uh.x.array.size]

    # Split mixed field; w is sub(0)
    w_h = uh.sub(0).collapse()

    # Build a VTK mesh for plotting scalar on cells/points
    topology, cell_types, geometry = plot.vtk_mesh(w_h.function_space)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Attach scalar field (w) at points
    grid.point_data["w"] = w_h.x.array.real

    # Warp geometry by w (out-of-plane). For 2D mesh, we create a pseudo z-warp.
    # PyVista warp_by_scalar uses the scalar as displacement along a normal; works for flat meshes.
    warped = grid.warp_by_scalar("w", factor=warp_scale)

    pl = pv.Plotter()
    pl.add_mesh(warped, scalars="w", show_edges=True)

    # Define the title based on the user's requirements
    # Formatting freq_mode as a float (e.g., .1f) if it's a number, otherwise just pass it through
    if isinstance(freq_mode, (float, int)):
        plot_name = f"mode_{bc}_{round(freq_mode, 1)}hz"
    else:
        plot_name = f"mode_{bc}_{freq_mode}hz"

    pl.add_title(plot_name)

    # Define filename with .png extension
    filename = f"plots/{plot_name}.png"

    # Passing screenshot=filename will save the image when the render window is closed
    pl.show(screenshot=filename)