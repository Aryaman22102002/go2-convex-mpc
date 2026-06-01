"""
wbc_qp.py
Whole-Body Control QP for stance leg torque computation.

Replaces the naive tau = J^T f mapping with a constrained optimization
that finds joint torques satisfying the rigid-body equations of motion
while keeping contact forces inside the friction cone and torques
within motor limits.

Decision variables:  z = [tau (12,), f_stance (3*n_s,)]
Cost:                min  Wf*||f - f*_MPC||^2  +  Wtau*||tau||^2
Constraints:         EOM (joint rows), friction pyramid, torque limits
"""

import numpy as np
import osqp
import scipy.sparse as sp

from convex_mpc.sim_params import MU, MU_SAFE

FZ_MIN = 10.0    # minimum normal force -- prevents solver from zeroing fz to trivially satisfy friction
WF     = 1.0     # weight on force tracking (primary objective)
W_TAU  = 1e-4    # weight on torque magnitude (regularization)

# Go2 motor limits per joint type (hip, thigh, calf)
TAU_MAX = np.array([23.7, 23.7, 45.43])

LEG_NAMES = ["FL", "FR", "RL", "RR"]
LEG_INDEX = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}


def solve_wbc(go2, contact_mask, mpc_forces):
    """
    Solve the WBC QP for all stance legs simultaneously.

    Parameters
    ----------
    go2          : PinGo2Model (already updated this tick via update_model)
    contact_mask : np.ndarray (4,)  -- 1 = stance, 0 = swing
    mpc_forces   : dict  leg_name -> np.ndarray(3,)  -- desired GRF from MPC

    Returns
    -------
    tau_out : dict  leg_name -> np.ndarray(3,)  -- commanded joint torques
    f_out   : dict  leg_name -> np.ndarray(3,)  -- solved contact forces
    success : bool
    """
    stance_legs = [name for name in LEG_NAMES if contact_mask[LEG_INDEX[name]] == 1]
    n_s = len(stance_legs)

    if n_s == 0:
        return ({leg: np.zeros(3) for leg in LEG_NAMES},
                {leg: np.zeros(3) for leg in LEG_NAMES},
                True)

    # ------------------------------------------------------------------
    # Rigid-body dynamics terms from Pinocchio
    # h = C*dq + g captures Coriolis, centripetal, and gravity effects.
    # We only need the actuated joint rows (6:18) since the floating base
    # has no motors and its rows would over-constrain f.
    # ------------------------------------------------------------------
    g_pin, C, M = go2.compute_dynamcis_terms()
    dq          = go2.current_config.get_dq()
    h_joints    = (C @ dq + g_pin)[6:18]    # (12,)

    # ------------------------------------------------------------------
    # Contact Jacobian -- joint rows only (12 x 3*n_s)
    # Each column maps a contact force component to a joint torque via J^T.
    # ------------------------------------------------------------------
    J_list = []
    f_ref  = np.zeros(3 * n_s)
    for k, leg in enumerate(stance_legs):
        J_full = go2.compute_full_foot_Jacobian_world(leg)  # (3 x 18)
        J_list.append(J_full[:, 6:18].T)                   # (12 x 3)
        f_ref[3*k:3*k+3] = mpc_forces[leg]
    Jc_joints = np.hstack(J_list)    # (12 x 3*n_s)

    n_tau = 12
    n_f   = 3 * n_s
    n_z   = n_tau + n_f
    i_f   = n_tau

    # ------------------------------------------------------------------
    # Cost matrix and linear term
    # P is diagonal -- WBC and torque weights on the respective blocks.
    # ------------------------------------------------------------------
    P_diag = np.concatenate([W_TAU * np.ones(n_tau), WF * np.ones(n_f)])
    P      = sp.diags(2 * P_diag, format='csc')
    q_vec  = np.zeros(n_z)
    q_vec[i_f:] = -2 * WF * f_ref    # linear term from expanding ||f - f_ref||^2

    # ------------------------------------------------------------------
    # EOM equality: tau + Jc^T f = h_joints
    # tau has 12 free variables, so this is always satisfiable regardless
    # of what f does -- no infeasibility from this constraint.
    # ------------------------------------------------------------------
    A_eq       = np.zeros((12, n_z))
    A_eq[:, :n_tau] = np.eye(12)
    A_eq[:, i_f:]   = Jc_joints
    b_eq            = h_joints

    # ------------------------------------------------------------------
    # Friction pyramid: linearized friction cone per stance foot.
    # MU_SAFE < MU gives a small safety margin so the solution stays
    # strictly inside the cone despite solver tolerance.
    # Row 5 enforces fz >= FZ_MIN to prevent degenerate zero-force solutions.
    # ------------------------------------------------------------------
    n_cone = 5 * n_s
    A_cone = np.zeros((n_cone, n_z))
    u_cone = np.zeros(n_cone)

    for k in range(n_s):
        r = 5 * k
        c = i_f + 3 * k
        A_cone[r+0, c:c+3] = [ 1,  0, -MU_SAFE]
        A_cone[r+1, c:c+3] = [-1,  0, -MU_SAFE]
        A_cone[r+2, c:c+3] = [ 0,  1, -MU_SAFE]
        A_cone[r+3, c:c+3] = [ 0, -1, -MU_SAFE]
        A_cone[r+4, c:c+3] = [ 0,  0, -1      ]
        u_cone[r+4] = -FZ_MIN

    # ------------------------------------------------------------------
    # Box constraints -- torque limits per joint, forces unconstrained.
    # Forces are bounded indirectly by the friction cone rows above.
    # ------------------------------------------------------------------
    tau_max_full = np.tile(TAU_MAX, 4)
    lb_box = np.concatenate([-tau_max_full, -1e9 * np.ones(n_f)])
    ub_box = np.concatenate([ tau_max_full,  1e9 * np.ones(n_f)])

    # stack everything into OSQP form:  l <= A z <= u
    A_total = sp.vstack([
        sp.csc_matrix(A_eq),
        sp.csc_matrix(A_cone),
        sp.eye(n_z, format='csc'),
    ], format='csc')
    l_total = np.concatenate([b_eq,  -1e9 * np.ones(n_cone), lb_box])
    u_total = np.concatenate([b_eq,   u_cone,                 ub_box])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    prob = osqp.OSQP()
    prob.setup(
        P, q_vec, A_total, l_total, u_total,
        warm_starting=True,
        verbose=False,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=2000,
        polish=True,
    )
    res = prob.solve()
    success = res.info.status in ("solved", "solved_inaccurate")

    # ------------------------------------------------------------------
    # Extract solution or fall back to naive J^T f if QP fails
    # ------------------------------------------------------------------
    tau_out = {leg: np.zeros(3) for leg in LEG_NAMES}
    f_out   = {leg: np.zeros(3) for leg in LEG_NAMES}

    if success and res.x is not None:
        tau_full = res.x[:n_tau]
        f_full   = res.x[i_f:]
        for k, leg in enumerate(stance_legs):
            tau_out[leg] = tau_full[LEG_INDEX[leg]*3 : LEG_INDEX[leg]*3+3]
            f_out[leg]   = f_full[3*k : 3*k+3]
    else:
        # QP failed -- fall back to direct Jacobian transpose mapping
        for leg in stance_legs:
            J3 = go2.compute_3x3_foot_Jacobian_world(leg)
            tau_out[leg] = J3.T @ -mpc_forces[leg]
            f_out[leg]   = mpc_forces[leg]
        print(f"[WBC] QP failed: {res.info.status}  n_s={n_s}")

    return tau_out, f_out, success
