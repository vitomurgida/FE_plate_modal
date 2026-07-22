import numpy as np
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from slepc4py import SLEPc


def plate_modal_shell_rm(
        l1, l2, t, E, nu, rho, num_modes,
        nx, ny,
        bc_type,  # "clamped", "simply_supported", or "simply_supported_free"
        target_sigma=1.0
):
    """
    Reissner–Mindlin plate modal analysis on a 2D midsurface mesh.
    Unknowns:
      - w: transverse displacement (scalar)
      - theta: rotations (2-vector)
    Eigenproblem:
      K x = lambda M x
      f = sqrt(lambda)/(2*pi)
    Boundary conditions:
      - clamped: w=0 and theta=(0,0) on boundary
      - simply_supported (practical): w=0 on boundary, theta free
        (implemented by constraining only the w-subspace DOFs)
      - simply_supported_free: w=0 on edges parallel to L2 (x=0, x=l1),
        edges parallel to L1 (y=0, y=l2) are completely free.
    """
    # --- Mesh (midsurface) ---
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([l1, l2])],
        [nx, ny],
        cell_type=mesh.CellType.triangle
    )

    # --- Function space: [w, theta] ---
    cell = domain.basix_cell()
    e_w = basix.ufl.element("Lagrange", cell, 2)
    e_th = basix.ufl.element("Lagrange", cell, 2, shape=(2,))
    e_mix = basix.ufl.mixed_element([e_w, e_th])
    V = fem.functionspace(domain, e_mix)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    w, theta = ufl.split(u)
    dw, dtheta = ufl.split(v)

    # --- Material ---
    G = E / (2.0 * (1.0 + nu))
    D = (E * t ** 3) / (12.0 * (1.0 - nu ** 2))
    k_s = 5.0 / 6.0  # shear correction factor

    # --- Kinematics ---
    kappa = ufl.sym(ufl.grad(theta))
    kappa_t = ufl.sym(ufl.grad(dtheta))

    gamma = ufl.grad(w) - theta
    gamma_t = ufl.grad(dw) - dtheta
    I = ufl.Identity(2)

    def moment(k):
        return D * ((1.0 - nu) * k + nu * ufl.tr(k) * I)

    # --- Bilinear forms ---
    a = ufl.inner(moment(kappa), kappa_t) * ufl.dx \
        + (k_s * G * t) * ufl.inner(gamma, gamma_t) * ufl.dx

    m = (rho * t) * (w * dw) * ufl.dx \
        + (rho * (t ** 3 / 12.0)) * ufl.inner(theta, dtheta) * ufl.dx

    # --- Boundary facets ---
    fdim = domain.topology.dim - 1

    def on_all_boundaries(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], l1) |
                np.isclose(x[1], 0.0) | np.isclose(x[1], l2))

    def on_l2_parallel_edges(x):
        # Edges parallel to L2 have constant x (x=0 and x=l1)
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], l1))

    def on_l1_parallel_edges(x):
        # Edges parallel to L1 have constant y (y=0 and y=l2)
        return (np.isclose(x[1], 0.0) | np.isclose(x[1], l2))

    # --- Dirichlet BCs and constrained dofs list for elimination ---
    if bc_type == "clamped":
        facets = mesh.locate_entities_boundary(domain, fdim, on_all_boundaries)
        # Clamp all DOFs in the mixed space
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        u0 = fem.Function(V)
        u0.x.array[:] = 0.0
        bc = fem.dirichletbc(u0, dofs)
        constrained_dofs = np.array(dofs, dtype=np.int32)

    elif bc_type == "simply_supported":
        facets = mesh.locate_entities_boundary(domain, fdim, on_all_boundaries)
        # Constrain only w = 0 on boundary.
        # Must collapse the subspace to create a Function for the BC value.
        Vw, submap = V.sub(0).collapse()  # Vw is a standalone space; submap maps Vw dofs -> V dofs
        dofs_w = fem.locate_dofs_topological(Vw, fdim, facets)
        w0 = fem.Function(Vw)
        w0.x.array[:] = 0.0
        bc = fem.dirichletbc(w0, dofs_w)
        # Map constrained dofs from collapsed subspace to parent mixed-space dofs
        constrained_dofs = np.array(submap[dofs_w], dtype=np.int32)

    elif bc_type == "simply_supported_free":
        facets = mesh.locate_entities_boundary(domain, fdim, on_l1_parallel_edges)
        # Constrain w = 0 ONLY on edges parallel to L2. Edges parallel to L1 remain free.
        Vw, submap = V.sub(0).collapse()
        dofs_w = fem.locate_dofs_topological(Vw, fdim, facets)
        w0 = fem.Function(Vw)
        w0.x.array[:] = 0.0
        bc = fem.dirichletbc(w0, dofs_w)
        constrained_dofs = np.array(submap[dofs_w], dtype=np.int32)

    else:
        raise ValueError("bc_type must be 'clamped', 'simply_supported', or 'simply_supported_free'.")

    # --- Assemble matrices ---
    K = assemble_matrix(fem.form(a))
    K.assemble()
    M = assemble_matrix(fem.form(m))
    M.assemble()

    # --- Enforce eigenproblem BCs by elimination ---
    # K: constrained rows/cols -> identity
    # M: constrained rows/cols -> zero
    K.zeroRowsColumns(constrained_dofs, diag=1.0)
    M.zeroRowsColumns(constrained_dofs, diag=0.0)

    # --- Eigen solve (SLEPc): Krylov-Schur + shift-and-invert ---
    eps = SLEPc.EPS().create(domain.comm)
    eps.setOperators(K, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target_sigma)

    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setDimensions(nev=num_modes)
    eps.setTolerances(tol=1e-9)
    eps.solve()

    nconv = eps.getConverged()
    freqs = []
    for i in range(min(num_modes, nconv)):
        lam = eps.getEigenvalue(i).real
        freqs.append(np.sqrt(abs(lam)) / (2.0 * np.pi))

    # --- Formatted Minimalist Output ---
    print("\nResonance Frequencies (Hz)")
    print("-" * 26)
    for i, f in enumerate(freqs, start=1):
        print(f"Mode {i:<2}: {f:>12.4f}")
    print()

    return freqs, eps, V