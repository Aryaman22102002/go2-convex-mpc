"""
go2_terrain_env.py
Gymnasium environment for Go2 quadruped terrain-adaptive locomotion.

Designed for PPO training with Stable-Baselines3.
No MPC or WBC during training -- pure RL from proprioception to joint targets.
WBC is added as a safety layer at inference time only.

Terrain curriculum:
    Phase 1 (0-1M steps):   flat only
    Phase 2 (1-2M steps):   flat + slope +-5 deg
    Phase 3 (2-4M steps):   flat + slope +-10 deg + step +-5cm
    Phase 4 (4-5M steps):   flat + slope +-15 deg + step +-8cm

Observation (56 dims):
    base roll, pitch          (2)
    base angular velocity     (3)
    base linear velocity      (3)
    joint angles              (12)
    joint velocities          (12)
    previous action           (12)
    gait phase sin/cos        (2)
    gravity vector in body    (3)  -- helps infer slope without terrain info
    foot contact flags        (4)  -- which feet are on ground
    base height               (1)  -- absolute height above ground
    
Action (12 dims):
    joint position targets -- PD controller converts to torques

Termination:
    base height < 0.15m (fallen)
    |roll| or |pitch| > 0.8 rad
    episode length > 1000 steps (5s at 200Hz)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from pathlib import Path
import xml.etree.ElementTree as ET
import tempfile
import os

# Load MPC reference trajectories for imitation reward
_REF_PATH = Path(__file__).resolve().parent / "data" / "mpc_reference.npz"
_REF = None
_REF_Q_FWD = None       # (2000, 12) joint angles for trot_forward
_REF_DQ_FWD = None      # (2000, 12) joint velocities for trot_forward
_REF_MASK_FWD = None    # (2000, 4)  foot contact mask for trot_forward
_REF_PHASE_FWD = None   # (2000,) phase values
_REF_BASE_QUAT_FWD = None   # (2000, 4) base orientation, (x,y,z,w) order
_REF_BASE_HEIGHT_FWD = None # (2000,) base z height
_REF_LINVEL_FWD = None      # (2000, 3) base linear velocity, finite-differenced
_REF_ANGVEL_FWD = None      # (2000, 3) base angular velocity, finite-differenced

REF_HZ = 200.0
CONTROL_HZ = 50.0
REF_FRAMES_PER_CONTROL_STEP = int(REF_HZ / CONTROL_HZ)  # 4 -- verified via diag_ref_rate.py

def _load_reference():
    global _REF, _REF_Q_FWD, _REF_DQ_FWD, _REF_MASK_FWD, _REF_PHASE_FWD
    global _REF_BASE_QUAT_FWD, _REF_BASE_HEIGHT_FWD, _REF_LINVEL_FWD, _REF_ANGVEL_FWD
    if _REF is not None:
        return
    if not _REF_PATH.exists():
        print(f"[WARNING] MPC reference not found at {_REF_PATH}. Imitation reward disabled.")
        return
    _REF = np.load(_REF_PATH)
    _REF_Q_FWD     = _REF["trot_forward_q"]       # (2000, 12)
    _REF_DQ_FWD    = _REF["trot_forward_dq"]      # (2000, 12)
    _REF_MASK_FWD  = _REF["trot_forward_mask"]    # (2000, 4)
    _REF_PHASE_FWD = _REF["trot_forward_phase"]   # (2000,)

    base_pos  = _REF["trot_forward_base_pos"]     # (2000, 3)
    base_quat = _REF["trot_forward_base_quat"]    # (2000, 4), (x,y,z,w) order -- verified
    _REF_BASE_QUAT_FWD   = base_quat
    _REF_BASE_HEIGHT_FWD = base_pos[:, 2]

    ref_dt = 1.0 / REF_HZ
    N = base_pos.shape[0]

    lin_vel = np.zeros((N, 3))
    lin_vel[1:-1] = (base_pos[2:] - base_pos[:-2]) / (2 * ref_dt)
    lin_vel[0]  = (base_pos[1] - base_pos[0]) / ref_dt
    lin_vel[-1] = (base_pos[-1] - base_pos[-2]) / ref_dt
    _REF_LINVEL_FWD = lin_vel

    def quat_to_rpy(qq):
        qx, qy, qz, qw = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        return roll, pitch, yaw

    roll, pitch, yaw = quat_to_rpy(base_quat)
    ang_vel = np.zeros((N, 3))
    for arr, col in [(roll, 0), (pitch, 1), (yaw, 2)]:
        dd = np.zeros(N)
        dd[1:-1] = (arr[2:] - arr[:-2]) / (2 * ref_dt)
        dd[0]  = (arr[1] - arr[0]) / ref_dt
        dd[-1] = (arr[-1] - arr[-2]) / ref_dt
        ang_vel[:, col] = dd
    _REF_ANGVEL_FWD = ang_vel

    print(f"[INFO] MPC reference loaded: {_REF_Q_FWD.shape[0]} frames "
          f"(q, dq, contact mask, base pose/velocity)")


_load_reference()


# PD gains for joint position control
KP = np.array([400.0, 400.0, 400.0,
               400.0, 400.0, 400.0,
               400.0, 400.0, 400.0,
               400.0, 400.0, 400.0])

KD = np.array([75.0, 75.0, 75.0,
               75.0, 75.0, 75.0,
               75.0, 75.0, 75.0,
               75.0, 75.0, 75.0])

TAU_MAX = np.array([23.7, 23.7, 45.43] * 4)

# qpos[7:] joint order: FL, FR, RL, RR
# ctrl actuator order:  FR, FL, RR, RL
# This reindex maps qpos joints to actuator slots
JOINT_REORDER = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
INV_JOINT_REORDER = np.argsort(JOINT_REORDER)  # converts actuator order -> qpos local order

# Default standing joint angles
Q_STAND = np.array([0.0, 0.9, -1.8,   # FL
                    0.0, 0.9, -1.8,   # FR
                    0.0, 0.9, -1.8,   # RL
                    0.0, 0.9, -1.8])  # RR

# Joint limits (approximate for Go2)
Q_MIN = np.array([-0.8, -1.5, -2.7] * 4)
Q_MAX = np.array([ 0.8,  3.4, -0.9] * 4)

# Gait parameters for phase clock
GAIT_HZ   = 3.0
GAIT_DUTY = 0.6

# Curriculum thresholds (total env steps)
CURRICULUM = [
    (0,          {"flat": 1.0, "slope": 0.0, "step": 0.0, "slope_max": 0.0,  "step_max": 0.0  }),
    (1_000_000,  {"flat": 0.4, "slope": 0.3, "step": 0.3, "slope_max": 8.0,  "step_max": 0.05 }),
    (3_000_000,  {"flat": 0.2, "slope": 0.4, "step": 0.4, "slope_max": 15.0, "step_max": 0.08 }),
]


class Go2TerrainEnv(gym.Env):
    """
    Go2 quadruped terrain environment for PPO training.
    Supports flat ground, slopes, and steps with curriculum learning.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, xml_path=None, render_mode=None,
                 ctrl_hz=50, sim_hz=1000,
                 total_steps_ref=None, steps_per_tick=1):
        super().__init__()

        self.render_mode  = render_mode
        self.ctrl_hz      = ctrl_hz        # policy frequency
        self.sim_hz       = sim_hz
        self.ctrl_decim   = sim_hz // ctrl_hz
        self.dt_ctrl      = 1.0 / ctrl_hz
        self.max_steps    = int(10.0 * ctrl_hz)  # 10 second episodes

        # Curriculum step counter. total_steps_ref is the starting offset
        # (the model's true cumulative trained steps at the time this env was
        # created -- previously always 0, even on --resume, so curriculum
        # silently restarted from scratch every training call). steps_per_tick
        # accounts for running n_envs parallel environments: the model's real
        # total_timesteps advances by n_envs every synchronized rollout step,
        # but each individual env instance only sees its own local step() calls
        # -- so each local increment should count as n_envs steps of true
        # progress, or this env's curriculum will keep under-counting relative
        # to the model's actual training progress for the rest of the session.
        self._total_steps   = 0 if total_steps_ref is None else total_steps_ref
        self._steps_per_tick = steps_per_tick

        # Find XML
        if xml_path is None:
            # assume running from repo root
            xml_path = Path(__file__).resolve().parents[1] / \
                       "models" / "MJCF" / "go2" / "scene.xml"
        self.xml_path = Path(xml_path)

        # Load base model to get dims
        self._model_flat = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self._data_flat  = mujoco.MjData(self._model_flat)

        self.n_joints = 12   # actuated joints
        self.n_obs    = 2 + 3 + 3 + 12 + 12 + 12 + 2 + 3 + 4 + 1  # = 54

        # Observation and action spaces
        obs_high = np.full(self.n_obs, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        act_high = np.ones(self.n_joints, dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)

        # Runtime state
        self.model  = None
        self.data   = None
        self._prev_action   = np.zeros(self.n_joints)
        self._step_count    = 0
        self._phase         = 0.0
        self._terrain_type  = "flat"
        self._episode_reward = 0.0

        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self._model_flat, height=480, width=640)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def _get_curriculum(self):
        steps = self._total_steps
        cfg = CURRICULUM[0][1]
        for threshold, c in CURRICULUM:
            if steps >= threshold:
                cfg = c
        return cfg

    # ------------------------------------------------------------------
    # Terrain XML generation
    # ------------------------------------------------------------------
    def _make_terrain_xml(self, terrain_type, slope_deg=0.0, step_height=0.0):
        """
        Returns path to a temporary XML file with the requested terrain.
        For flat ground, returns the original scene.xml path.
        """
        if terrain_type == "flat":
            return str(self.xml_path)

        # Parse base scene XML
        tree = ET.parse(str(self.xml_path))
        root = tree.getroot()

        worldbody = root.find("worldbody")

        # Remove existing floor geom
        for geom in worldbody.findall("geom"):
            if geom.get("name") == "floor" or geom.get("type") == "plane":
                worldbody.remove(geom)

        if terrain_type == "slope":
            # Tilted plane
            slope_rad = np.deg2rad(slope_deg)
            # euler: rotate around Y axis
            euler = f"0 {slope_deg} 0"
            ET.SubElement(worldbody, "geom", {
                "name":     "floor",
                "type":     "plane",
                "euler":    euler,
                "size":     "10 10 0.1",
                "material": "groundplane",
                "friction": "0.8 0.005 0.0001",
            })

        elif terrain_type == "step_up":
            # Flat ground then a raised platform
            ET.SubElement(worldbody, "geom", {
                "name":     "floor",
                "type":     "plane",
                "size":     "10 10 0.1",
                "material": "groundplane",
                "friction": "0.8 0.005 0.0001",
            })
            # Step platform starting at x=1.0
            ET.SubElement(worldbody, "geom", {
                "name":     "step",
                "type":     "box",
                "pos":      f"2.0 0 {step_height/2:.3f}",
                "size":     f"2.0 2.0 {step_height/2:.3f}",
                "rgba":     "0.6 0.4 0.2 1",
                "friction": "0.8 0.005 0.0001",
            })

        elif terrain_type == "step_down":
            # Raised platform then a drop
            ET.SubElement(worldbody, "geom", {
                "name":     "floor_low",
                "type":     "plane",
                "pos":      f"3.0 0 {-step_height:.3f}",
                "size":     "10 10 0.1",
                "material": "groundplane",
                "friction": "0.8 0.005 0.0001",
            })
            ET.SubElement(worldbody, "geom", {
                "name":     "platform",
                "type":     "box",
                "pos":      f"-1.0 0 {-step_height/2:.3f}",
                "size":     f"3.0 2.0 {step_height/2:.3f}",
                "rgba":     "0.6 0.4 0.2 1",
                "friction": "0.8 0.005 0.0001",
            })

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(
            suffix=".xml", delete=False,
            dir=str(self.xml_path.parent)   # same dir so relative paths work
        )
        tree.write(tmp.name)
        tmp.close()
        return tmp.name

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Multi-terrain set aside for now -- focus is on getting a solid
        # walking gait first. Forcing flat regardless of curriculum stage.
        self._terrain_type = "flat"
        xml_to_load = str(self.xml_path)

        # Load model
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_to_load)
        except Exception:
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self._terrain_type = "flat"

        self.model.opt.timestep = 1.0 / self.sim_hz
        self.data = mujoco.MjData(self.model)

        # ------------------------------------------------------------
        # Reference State Initialization (RSI): instead of always starting
        # from a standing pose, start each episode with the robot's actual
        # physical state (position, orientation, joint angles, all
        # velocities) set directly from a RANDOM point along the recorded
        # MPC reference gait cycle. Combined with a fully deterministic
        # phase clock (see step()), this replaces the entire adaptive
        # nearest-neighbor matching system that caused every feedback-loop
        # problem found during debugging (contact-dependent phase
        # selection, circular clearance gating, cadence drift). There is no
        # matching left to game: the target at any moment is simply
        # "wherever the reference would be if followed correctly since a
        # known starting point."
        # ------------------------------------------------------------
        if _REF_Q_FWD is not None:
            n_ref = _REF_Q_FWD.shape[0]
            ref_start_idx = int(rng.integers(0, n_ref))
            self._ref_start_idx = ref_start_idx

            ref_q_local  = _REF_Q_FWD[ref_start_idx][INV_JOINT_REORDER]
            ref_dq_local = _REF_DQ_FWD[ref_start_idx][INV_JOINT_REORDER]
            ref_quat_xyzw = _REF_BASE_QUAT_FWD[ref_start_idx]
            ref_height    = _REF_BASE_HEIGHT_FWD[ref_start_idx]
            ref_linvel    = _REF_LINVEL_FWD[ref_start_idx]
            ref_angvel    = _REF_ANGVEL_FWD[ref_start_idx]

            self.data.qpos[0]   = 0.0   # reset episode-local x/y origin each time
            self.data.qpos[1]   = 0.0
            self.data.qpos[2]   = ref_height
            # MuJoCo qpos quat order is (w,x,y,z); reference is (x,y,z,w)
            qx, qy, qz, qw = ref_quat_xyzw
            self.data.qpos[3:7] = [qw, qx, qy, qz]
            self.data.qpos[7:]  = ref_q_local

            self.data.qvel[0:3] = ref_linvel
            self.data.qvel[3:6] = ref_angvel
            self.data.qvel[6:]  = ref_dq_local
        else:
            self._ref_start_idx = 0
            self.data.qpos[:3]  = [0.0, 0.0, 0.35]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qpos[7:]  = Q_STAND + rng.uniform(-0.05, 0.05, 12)
            self.data.qvel[:]   = rng.uniform(-0.05, 0.05, self.model.nv)

        self._episode_step = 0
        self._last_ref_idx = self._ref_start_idx

        mujoco.mj_forward(self.model, self.data)

        # Real starting yaw from whatever orientation RSI actually set (not
        # hardcoded 0 -- RSI can start mid-stride at a non-identity heading)
        qw, qx, qy, qz = self.data.qpos[3], self.data.qpos[4], self.data.qpos[5], self.data.qpos[6]
        self._init_yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

        self._prev_action  = np.zeros(self.n_joints)
        self._step_count   = 0
        self._episode_reward = 0.0
        self._foot_state_duration = np.zeros(4)  # consecutive steps in current contact state, per foot [FL,FR,RL,RR]
        self._foot_prev_contact   = None          # previous step's contact bools, to detect state flips

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        # Scale action from [-1, 1] to joint position targets
        q_target = Q_STAND + action * 0.5   # +-0.5 rad around stand pose
        q_target = np.clip(q_target, Q_MIN, Q_MAX)

        # PD control
        q_now  = self.data.qpos[7:19]
        dq_now = self.data.qvel[6:18]
        tau    = KP * (q_target - q_now) - KD * dq_now
        tau    = np.clip(tau, -TAU_MAX, TAU_MAX)

        # Simulate at sim_hz
        for _ in range(self.ctrl_decim):
            q_now  = self.data.qpos[7:19][JOINT_REORDER]
            dq_now = self.data.qvel[6:18][JOINT_REORDER]
            tau    = KP * (q_target - q_now) - KD * dq_now
            tau    = np.clip(tau, -TAU_MAX, TAU_MAX)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        # Deterministic reference index (RSI + fixed-rate advance -- see
        # reset() and the reward function for why this replaced the old
        # adaptive nearest-neighbor matching entirely)
        self._episode_step += 1
        if _REF_Q_FWD is not None:
            n_ref = _REF_Q_FWD.shape[0]
            self._last_ref_idx = (self._ref_start_idx +
                                   self._episode_step * REF_FRAMES_PER_CONTROL_STEP) % n_ref
        else:
            self._last_ref_idx = None

        self._step_count  += 1
        self._total_steps += self._steps_per_tick

        obs     = self._get_obs()
        reward  = self._compute_reward(action, tau)
        terminated = self._is_terminated()
        truncated  = self._step_count >= self.max_steps

        self._prev_action    = action.copy()
        self._episode_reward += reward

        info = {}
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._step_count,
                "terrain": self._terrain_type,
            }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        d = self.data

        # Base orientation as roll/pitch
        qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        # actually MuJoCo stores qw first in free joint
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))

        # Gravity vector in body frame (tells the policy about slope)
        R = np.zeros((3,3))
        mujoco.mju_quat2Mat(R.ravel(), d.qpos[3:7])
        g_world = np.array([0, 0, -1.0])
        g_body  = R.T @ g_world

        # Base velocities
        lin_vel = d.qvel[0:3].copy()
        ang_vel = d.qvel[3:6].copy()

        # Joint state -- reordered to match actuator convention (FR,FL,RR,RL)
        q_joints  = d.qpos[7:19][JOINT_REORDER].copy()
        dq_joints = d.qvel[6:18][JOINT_REORDER].copy()

        # Gait phase -- derived from the deterministic reference index
        # (self._last_ref_idx, set in reset()/step() from Reference State
        # Initialization + a fixed advance rate). This is always exactly
        # correct and lag-free now, unlike the old adaptive nearest-neighbor
        # match it replaced.
        if self._last_ref_idx is not None and _REF_PHASE_FWD is not None:
            matched_phase = _REF_PHASE_FWD[self._last_ref_idx]
        else:
            matched_phase = 0.0
        phase_sin = np.sin(matched_phase)
        phase_cos = np.cos(matched_phase)

        # Foot contact flags (simple height check)
        foot_names = ["FL_foot", "FR_foot",
                      "RL_foot", "RR_foot"]
        contact_flags = np.zeros(4)
        for i, fname in enumerate(foot_names):
            try:
                fid = self.model.body(fname).id
                foot_z = d.xpos[fid, 2]
                contact_flags[i] = float(foot_z < 0.05)
            except Exception:
                contact_flags[i] = 0.0

        # Base height
        base_height = np.array([d.qpos[2]])

        obs = np.concatenate([
            [roll, pitch],        # 2
            ang_vel,              # 3
            lin_vel,              # 3
            q_joints,             # 12
            dq_joints,            # 12
            self._prev_action,    # 12
            [phase_sin, phase_cos],  # 2
            g_body,               # 3
            contact_flags,        # 4
            base_height,          # 1
        ]).astype(np.float32)

        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action, tau):
        d = self.data

        # Orientation
        qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        r_upright = np.exp(-5.0 * (roll**2 + pitch**2))

        # Yaw stability -- penalize drift from initial heading (fix: policy was spiraling)
        yaw_err = np.arctan2(np.sin(yaw - self._init_yaw), np.cos(yaw - self._init_yaw))
        # Yaw stability -- weight increased substantially (0.5 -> 4.0): at a
        # typical observed yaw drift of ~0.15 rad, the old weight cost only
        # -0.011 in reward, negligible next to imitation/clearance terms
        # (0.1-1+ scale). Confirmed via diagnostics that rear-leg high-lift
        # events (a REAL characteristic inherited from the reference gait,
        # not a bug -- the MPC reference itself has rear feet swinging
        # ~20-25% higher than front feet) correlate strongly with yaw drift
        # the policy isn't adequately correcting for, unlike the real
        # MPC controller's closed-loop feedback.
        r_yaw = -4.0 * yaw_err**2

        # Height
        height_err = d.qpos[2] - 0.27
        r_height = np.exp(-10.0 * height_err**2)

        # Forward velocity -- capped at 0.6 m/s to match reference speed
        # (fix: was uncapped at 2.0, causing policy to outrun the reference gait)
        vx = d.qvel[0]
        r_forward = np.clip(vx, 0.0, 0.6) * r_upright * 0.3

        # Lateral velocity penalty
        vy = d.qvel[1]
        r_lateral = -0.3 * vy**2

        # Torque penalty
        r_torque = -1e-5 * np.sum(tau**2)

        # Action smoothness
        r_smooth = -0.01 * np.sum((action - self._prev_action)**2)

        # Foot contact detection (clearance reward computed further below,
        # after the reference match -- see note there for why)
        foot_names = ["FL_foot", "FR_foot",
                      "RL_foot", "RR_foot"]
        contacts_now = np.zeros(4)
        foot_heights = np.zeros(4)
        for i, fname in enumerate(foot_names):
            try:
                fid = self.model.body(fname).id
                foot_z = d.xpos[fid, 2]
                foot_heights[i] = foot_z
                contacts_now[i] = 1.0 if foot_z < 0.05 else 0.0
            except Exception:
                pass

        # Diagonal synchrony (explicit, direct signal for correct trot timing --
        # complements the imitation/clearance rewards above, which only imply
        # correct diagonal pairing indirectly through joint-angle/contact-mask
        # matching. This directly rewards the real contact pattern for being
        # close to either valid diagonal-trot configuration: FL+RR swinging
        # while FR+RL stand, or the reverse. contacts_now order is
        # [FL, FR, RL, RR]; 1=stance, 0=swing.)
        pattern_a = np.array([0.0, 1.0, 1.0, 0.0])  # FL+RR swing, FR+RL stance
        pattern_b = np.array([1.0, 0.0, 0.0, 1.0])  # FR+RL swing, FL+RR stance
        dist_a = np.sum(np.abs(contacts_now - pattern_a))
        dist_b = np.sum(np.abs(contacts_now - pattern_b))
        r_sync = 1.0 - 0.25 * min(dist_a, dist_b)  # 1.0 if exactly matching either pattern

        # Stuck-leg penalty (fix: both the clearance-gate and r_sync above
        # turned out to be defeatable -- the clearance gate was circular
        # (the reference match is partly CHOSEN to agree with real contacts,
        # since mask_dists is part of the matching cost, so "reference
        # expectation" just rubber-stamps whatever the robot is already
        # doing), and r_sync only checks the INSTANTANEOUS 4-foot pattern,
        # which a permanently-fixed pair (e.g. RL always stance, RR always
        # swing) can still satisfy as long as the OTHER two feet alternate
        # normally. This penalty is direct and non-circular: track real
        # elapsed steps each foot has spent continuously in its current
        # state, and penalize hard once any foot exceeds a reasonable bound
        # -- independent of any reference match or other feet's behavior, so
        # no single foot can hide in one state indefinitely.)
        if self._foot_prev_contact is None:
            self._foot_prev_contact = contacts_now.copy()
        state_changed = (contacts_now != self._foot_prev_contact)
        self._foot_state_duration = np.where(state_changed, 0, self._foot_state_duration + 1)
        self._foot_prev_contact = contacts_now.copy()

        MAX_STATE_DURATION = 20  # control steps (~0.4s at 50Hz) -- generous vs.
                                 # reference's own ~8-step typical swing/stance
        r_stuck_leg = 0.0
        for i in range(4):
            if self._foot_state_duration[i] > MAX_STATE_DURATION:
                overage = self._foot_state_duration[i] - MAX_STATE_DURATION
                r_stuck_leg -= 0.05 * overage


        # Imitation reward -- direct lookup against the DETERMINISTIC
        # reference index (self._last_ref_idx, set in step() from Reference
        # State Initialization + a fixed 4-frames-per-control-step advance
        # rate -- see reset() for the full rationale). This replaced an
        # adaptive nearest-neighbor matching system that searched for
        # whichever reference frame the robot's current state most closely
        # resembled, weighted by joint angle, velocity, and contact pattern
        # similarity, constrained to a small window to prevent teleporting.
        # That system caused a cascade of feedback-loop bugs during
        # debugging: it could match a frame with a completely wrong contact
        # pattern; its "reference expectation" was partly CHOSEN to agree
        # with real contacts (since contact-mask distance was part of the
        # matching cost), making downstream gating on that expectation
        # circular; and it let the policy control the pace of its own
        # imitation target rather than being held to a fixed pace. None of
        # that is possible anymore: the target is now simply "wherever the
        # reference would be if followed correctly since a known random
        # starting point," with no dependence on the robot's real state at
        # all. This is the standard DeepMimic-style design.
        r_imitate = 0.0
        r_vel_match = 0.0
        ref_idx = self._last_ref_idx
        if _REF_Q_FWD is not None and ref_idx is not None:
            q_now  = d.qpos[7:19][JOINT_REORDER]
            dq_now = d.qvel[6:18][JOINT_REORDER]

            ref_q  = _REF_Q_FWD[ref_idx]
            ref_dq = _REF_DQ_FWD[ref_idx]

            joint_err = np.sum((q_now - ref_q)**2)
            r_imitate = np.exp(-2.0 * joint_err)

            vel_err = np.sum((dq_now - ref_dq)**2)
            r_vel_match = np.exp(-0.1 * vel_err)

        # Foot clearance (fix: measured actual swing height on v10 was only
        # ~0.06-0.09m vs. the reference's own ~0.13-0.16m, so this weight was
        # raised substantially -- but that alone created a new, worse exploit:
        # since clearance was rewarded for ANY foot off the ground regardless
        # of gait phase, a leg that simply never lands collects this reward
        # every single step forever, which is a bigger and more reliable
        # payoff than a real trot's swing phase (only ~30-50% duty cycle) --
        # confirmed empirically: one leg (RR) went permanently airborne while
        # its diagonal partner (RL) stayed permanently planted to compensate.
        # Fix: only reward clearance for a foot when the MATCHED REFERENCE
        # FRAME also expects that foot to be swinging right now (its own
        # recorded mask says not-in-contact) -- this ties "reward for
        # lifting" to the actual gait cycle instead of an open-ended
        # "any air time is good" signal that a permanently-lifted leg could
        # exploit indefinitely.
        r_clearance = 0.0
        if _REF_Q_FWD is not None and ref_idx is not None:
            expected_mask = _REF_MASK_FWD[ref_idx]  # 1=stance, 0=swing, per reference
            for i in range(4):
                if expected_mask[i] < 0.5 and contacts_now[i] < 0.5:
                    # reference expects swing AND foot is actually off the ground
                    r_clearance += np.clip(foot_heights[i], 0, 0.16) * 3.0
        else:
            # no reference available -- fall back to the old unconditional
            # behavior rather than giving zero clearance signal
            for i in range(4):
                if contacts_now[i] < 0.5:
                    r_clearance += np.clip(foot_heights[i], 0, 0.16) * 3.0

        self._last_reward_components = {
            "r_upright": r_upright, "r_yaw": r_yaw, "r_height": r_height,
            "r_forward": r_forward, "r_lateral": r_lateral, "r_torque": r_torque,
            "r_smooth": r_smooth, "r_clearance": r_clearance,
            "r_imitate": r_imitate, "r_vel_match": r_vel_match, "r_sync": r_sync,
            "r_stuck_leg": r_stuck_leg,
            "contacts_now": contacts_now.copy(),
            "ref_mask": (_REF_MASK_FWD[ref_idx].copy() if _REF_Q_FWD is not None else None),
        }

        return float(r_upright + r_yaw + r_height + r_forward + r_lateral +
                     r_torque + r_smooth + r_clearance + r_imitate + r_vel_match +
                     r_sync + r_stuck_leg)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _is_terminated(self):
        d = self.data

        # Fallen (too low)
        if d.qpos[2] < 0.15:
            return True

        # Extreme tilt
        qw, qx, qy, qz = d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        if abs(roll) > 0.8 or abs(pitch) > 0.8:
            return True

        return False

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array" and self.data is not None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        pass
