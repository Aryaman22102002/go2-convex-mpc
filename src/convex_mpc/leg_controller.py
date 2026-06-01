import time
import numpy as np
from .go2_robot_data import PinGo2Model
from .gait import Gait
from .wbc_qp import solve_wbc
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------
# Leg Controller Setting
# --------------------------------------------------------------------------------

KP_SWING = np.diag([400, 400, 400])
KD_SWING = np.diag([75, 75, 75])

from convex_mpc.sim_params import MU  # Friction coefficient
TAU_LIM_PER_JOINT = np.array([23.7, 23.7, 45.43])  # hip, thigh, calf (Nm)

# Mapping from leg name to index in the mask
LEG_INDEX = {
    "FL": 0,
    "FR": 1,
    "RL": 2,
    "RR": 3,
}

LEG_NAMES = ["FL", "FR", "RL", "RR"]

# Mapping from leg name to the joint torque slice in (C*dq + g)
JOINT_SLICES = {
    "FL": slice(6, 9),
    "FR": slice(9, 12),
    "RL": slice(12, 15),
    "RR": slice(15, 18),
}

@dataclass
class LegOutput:
    tau: np.ndarray         # shape (3,)
    pos_des: np.ndarray     # shape (3,)
    pos_now: np.ndarray     # shape (3,)
    vel_des: np.ndarray     # shape (3,)
    vel_now: np.ndarray     # shape (3,)
    # Violation metrics (positive = violation, negative = safe margin)
    fx_viol: float = 0.0    # |fx| - mu*fz  (friction x)
    fy_viol: float = 0.0    # |fy| - mu*fz  (friction y)
    fz_viol: float = 0.0    # -fz           (pulling ground)
    tau_viol: float = 0.0   # max(|tau| - tau_limit_per_joint)
    is_stance: bool = False


class LegController():

    def __init__(self):
        self.last_mask = np.array([2, 2, 2, 2])

    # ------------------------------------------------------------------
    # Solve WBC once for all legs, return all outputs
    # ------------------------------------------------------------------
    def compute_all_torques(
        self,
        go2: PinGo2Model,
        gait: Gait,
        mpc_forces_world: np.ndarray,   # shape (12,): [FL, FR, RL, RR] x [fx,fy,fz]
        current_time: float,
        use_wbc: bool = True,
    ) -> dict:
        """
        Compute torques for all four legs.
        Stance legs: solved via WBC QP (if use_wbc=True) or naive JᵀF.
        Swing legs:  impedance controller (unchanged).

        Returns dict: leg_name -> LegOutput
        """
        current_mask = gait.compute_current_mask(current_time)  # (4,)

        # Split MPC forces into per-leg dict
        mpc_force_dict = {
            "FL": mpc_forces_world[0:3],
            "FR": mpc_forces_world[3:6],
            "RL": mpc_forces_world[6:9],
            "RR": mpc_forces_world[9:12],
        }

        # ---- WBC solve for all stance legs at once ----
        self.wbc_solve_time_ms = 0.0
        if use_wbc:
            t0 = time.perf_counter()
            wbc_tau, wbc_f, wbc_ok = solve_wbc(go2, current_mask, mpc_force_dict)
            self.wbc_solve_time_ms = (time.perf_counter() - t0) * 1e3
            if not wbc_ok:
                print(f"[LEG] WBC fallback at t={current_time:.3f}s  mask={current_mask}")
        else:
            wbc_tau = None
            wbc_f   = mpc_force_dict
            wbc_ok  = False

        outputs = {}
        for leg in LEG_NAMES:
            outputs[leg] = self._compute_single_leg(
                leg, go2, gait, current_mask,
                mpc_force_dict[leg],
                wbc_tau, wbc_f, wbc_ok,
                current_time,
            )

        # Update mask memory after all legs processed
        self.last_mask = current_mask.reshape(4,)

        return outputs

    # ------------------------------------------------------------------
    # Internal: compute one leg given pre-solved WBC result
    # ------------------------------------------------------------------
    def _compute_single_leg(
        self,
        leg: str,
        go2: PinGo2Model,
        gait: Gait,
        current_mask: np.ndarray,
        contact_force: np.ndarray,
        wbc_tau: dict,
        wbc_f: dict,
        wbc_ok: bool,
        current_time: float,
    ) -> LegOutput:

        leg_idx     = LEG_INDEX[leg]
        joint_slice = JOINT_SLICES[leg]

        J_foot_world      = go2.compute_3x3_foot_Jacobian_world(leg)      # (3x3)
        J_full_foot_world = go2.compute_full_foot_Jacobian_world(leg)      # (3x18)
        g, C, M           = go2.compute_dynamcis_terms()

        tau_cmd   = np.zeros(3)
        fx_viol   = 0.0
        fy_viol   = 0.0
        fz_viol   = 0.0
        tau_viol  = 0.0
        is_stance = False

        foot_pos_des, foot_vel_des = go2.get_single_foot_state_in_world(leg)
        foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

        # Detect takeoff transition
        if self.last_mask[leg_idx] != current_mask[leg_idx] and current_mask[leg_idx] == 0:
            setattr(self, f"{leg}_takeoff_time", current_time)
            traj, td_pos = gait.compute_swing_traj_and_touchdown(go2, leg)
            setattr(self, f"{leg}_traj", traj)
            setattr(self, f"{leg}_td_pos", td_pos)

        if current_mask[leg_idx] == 0:  # ---- Swing phase ----
            takeoff_time = getattr(self, f"{leg}_takeoff_time")
            traj         = getattr(self, f"{leg}_traj")

            time_since_takeoff = current_time - takeoff_time
            foot_pos_des, foot_vel_des, foot_acl_des = traj(time_since_takeoff)
            foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

            pos_error = foot_pos_des - foot_pos_now
            vel_error = foot_vel_des - foot_vel_now

            Lambda   = np.linalg.inv(J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T)
            Jdot_dq  = go2.compute_Jdot_dq_world(leg)
            f_ff     = Lambda @ (foot_acl_des - Jdot_dq)
            force    = KP_SWING @ pos_error + KD_SWING @ vel_error + f_ff

            tau_cmd  = J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[joint_slice]

        else:  # ---- Stance phase ----
            is_stance = True

            if wbc_ok and wbc_tau is not None:
                # Use WBC solution
                tau_cmd = wbc_tau[leg]
                f_used  = wbc_f[leg]
            else:
                # Fallback: naive JᵀF
                tau_cmd = J_foot_world.T @ -contact_force
                f_used  = contact_force

            # Violation logging against the force actually used
            f = f_used.reshape(3,)
            tau_flat = tau_cmd.reshape(3,)
            fx_viol  = float(abs(f[0]) - MU * f[2])
            fy_viol  = float(abs(f[1]) - MU * f[2])
            fz_viol  = float(-f[2])
            tau_viol = float(np.max(np.abs(tau_flat) - TAU_LIM_PER_JOINT))

        return LegOutput(
            tau      = tau_cmd.reshape(3,),
            pos_des  = foot_pos_des,
            pos_now  = foot_pos_now,
            vel_des  = foot_vel_des,
            vel_now  = foot_vel_now,
            fx_viol  = fx_viol,
            fy_viol  = fy_viol,
            fz_viol  = fz_viol,
            tau_viol = tau_viol,
            is_stance= is_stance,
        )

    # ------------------------------------------------------------------
    # Legacy per-leg method (kept for backward compatibility)
    # ------------------------------------------------------------------
    def compute_leg_torque(
        self,
        leg: str,
        go2: PinGo2Model,
        gait: Gait,
        contact_force: np.ndarray,
        current_time: float,
    ) -> LegOutput:
        current_mask = gait.compute_current_mask(current_time)
        J_foot_world      = go2.compute_3x3_foot_Jacobian_world(leg)
        J_full_foot_world = go2.compute_full_foot_Jacobian_world(leg)
        g, C, M           = go2.compute_dynamcis_terms()

        leg_idx     = LEG_INDEX[leg]
        joint_slice = JOINT_SLICES[leg]
        tau_cmd     = np.zeros(3)
        fx_viol = fy_viol = fz_viol = tau_viol = 0.0
        is_stance = False

        foot_pos_des, foot_vel_des = go2.get_single_foot_state_in_world(leg)
        foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)

        if self.last_mask[leg_idx] != current_mask[leg_idx] and current_mask[leg_idx] == 0:
            setattr(self, f"{leg}_takeoff_time", current_time)
            traj, td_pos = gait.compute_swing_traj_and_touchdown(go2, leg)
            setattr(self, f"{leg}_traj", traj)
            setattr(self, f"{leg}_td_pos", td_pos)

        if current_mask[leg_idx] == 0:
            takeoff_time = getattr(self, f"{leg}_takeoff_time")
            traj         = getattr(self, f"{leg}_traj")
            time_since_takeoff = current_time - takeoff_time
            foot_pos_des, foot_vel_des, foot_acl_des = traj(time_since_takeoff)
            foot_pos_now, foot_vel_now = go2.get_single_foot_state_in_world(leg)
            pos_error = foot_pos_des - foot_pos_now
            vel_error = foot_vel_des - foot_vel_now
            Lambda    = np.linalg.inv(J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T)
            Jdot_dq   = go2.compute_Jdot_dq_world(leg)
            f_ff      = Lambda @ (foot_acl_des - Jdot_dq)
            force     = KP_SWING @ pos_error + KD_SWING @ vel_error + f_ff
            tau_cmd   = J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[joint_slice]
        else:
            tau_cmd   = J_foot_world.T @ -contact_force
            is_stance = True
            f         = contact_force.reshape(3,)
            tau_flat  = tau_cmd.reshape(3,)
            fx_viol   = float(abs(f[0]) - MU * f[2])
            fy_viol   = float(abs(f[1]) - MU * f[2])
            fz_viol   = float(-f[2])
            tau_viol  = float(np.max(np.abs(tau_flat) - TAU_LIM_PER_JOINT))

        self.last_mask[leg_idx] = current_mask[leg_idx]

        return LegOutput(
            tau=tau_cmd.reshape(3,), pos_des=foot_pos_des, pos_now=foot_pos_now,
            vel_des=foot_vel_des, vel_now=foot_vel_now,
            fx_viol=fx_viol, fy_viol=fy_viol, fz_viol=fz_viol,
            tau_viol=tau_viol, is_stance=is_stance,
        )
