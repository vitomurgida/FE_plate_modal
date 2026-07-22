def animate_mode_shape_interactive_pyvista046(eps, V, mode, freq_mode, warp_scale, fps, save_video=True):
    """
    PyVista 0.46.3-compatible interactive animation with video export.
    Controls:
      - Key 'p': toggle Play/Pause
      - Key 's': Stop/reset to undeformed
      - Key 'q' or 'Escape': quit/close window
    Notes:
      - Uses pl.show(interactive_update=True, auto_close=False) + pl.update()
      - Automatically saves a 1-cycle MP4 video before starting interactive mode if save_video=True.
    """
    import time
    import numpy as np
    from petsc4py import PETSc
    import pyvista as pv
    from dolfinx import fem, plot

    i = mode - 1
    nconv = eps.getConverged()
    if i < 0 or i >= nconv:
        raise ValueError(f"Requested mode={mode}, but only {nconv} modes converged.")

    # --- Extract eigenvector into dolfinx Function ---
    uh = fem.Function(V)
    xr = PETSc.Vec().createMPI(V.dofmap.index_map.size_global, comm=V.mesh.comm)
    xi = PETSc.Vec().createMPI(V.dofmap.index_map.size_global, comm=V.mesh.comm)
    eps.getEigenvector(i, xr, xi)
    uh.x.array[:] = xr.getArray(readonly=True)[: uh.x.array.size]

    # transverse displacement subfield
    w_h = uh.sub(0).collapse()

    # --- Build VTK grid once ---
    topology, cell_types, geometry = plot.vtk_mesh(w_h.function_space)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    wvals = w_h.x.array.real.copy()
    grid.point_data["w"] = wvals

    # Ensure 3D points so we can warp in z
    pts0 = grid.points
    if pts0.shape[1] == 2:
        pts0 = np.column_stack([pts0, np.zeros(pts0.shape[0])])
        grid.points = pts0
    grid_warped = grid.copy(deep=True)
    pts_base = grid.points.copy()

    # --- State ---
    state = {"playing": False, "phase": 0.0, "running": True}
    dt = 1.0 / float(fps)

    # Visual oscillation speed:
    # 1 cycle per second -> omega_vis = 2*pi rad/s
    omega_vis = 2.0 * np.pi
    dphase = omega_vis * dt

    def apply_deformation():
        s = np.sin(state["phase"]) * warp_scale
        pts = pts_base.copy()
        pts[:, 2] += s * wvals
        grid_warped.points = pts

    # --- Plotter ---
    pl = pv.Plotter(window_size=[2000, 2000])
    pl.add_mesh(grid_warped, scalars="w", show_edges=True)
    pl.add_title(f"Mode {mode}, {round(freq_mode, 1)} Hz: keys p=play/pause, s=stop, q=quit")

    def toggle_play():
        state["playing"] = not state["playing"]

    def stop_reset():
        state["playing"] = False
        state["phase"] = 0.0
        grid_warped.points = pts_base

    def quit_close():
        state["running"] = False
        pl.close()

    pl.add_key_event("p", toggle_play)
    pl.add_key_event("s", stop_reset)
    pl.add_key_event("q", quit_close)
    pl.add_key_event("Escape", quit_close)

    # Show in interactive-update mode (critical)
    pl.show(interactive_update=True, auto_close=False)

    # =====================================================================
    # >>> NEW SECTION: SAVE COMPRESSED VIDEO <<<
    # =====================================================================
    if save_video:
        # MP4 format with H.264 codec yields the lowest memory usage for videos
        filename = f"mode_ss_{round(freq_mode,1)}hz.mp4"

        # quality=4 is chosen to prioritize low memory usage while keeping clean lines
        pl.open_movie(filename, framerate=fps, quality=10)

        # Record exactly 1 full cycle (1 second of animation at visual omega = 2*pi)
        frames_to_record = int(fps)
        for _ in range(frames_to_record):
            state["phase"] += dphase
            apply_deformation()
            pl.render()
            pl.write_frame()

        # Close the movie writer to finalize the file and flush from memory
        if hasattr(pl, 'mwriter') and pl.mwriter is not None:
            pl.mwriter.close()

        # Reset the mesh back to the undeformed state for the user's interactive session
        stop_reset()
        pl.update()
    # =====================================================================

    # GUI-friendly loop
    while state["running"]:
        if state["playing"]:
            state["phase"] += dphase
            apply_deformation()
        pl.update()
        time.sleep(dt)