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
_REF_Q_FWD = None    # (2000, 12) joint angles for trot_forward
_REF_PHASE_FWD = None  # (2000,) phase values

def _load_reference():
    global _REF, _REF_Q_FWD, _REF_PHASE_FWD
    if _REF is not None:
        return
    if not _REF_PATH.exists():
        print(f"[WARNING] MPC reference not found at {_REF_PATH}. Imitation reward disabled.")
        return
    _REF = np.load(_REF_PATH)
    _REF_Q_FWD     = _REF["trot_forward_q"]      # (2000, 12)
    _REF_PHASE_FWD = _REF["trot_forward_phase"]   # (2000,)
    print(f"[INFO] MPC reference loaded: {_REF_Q_FWD.shape[0]} frames")

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
                 total_steps_ref=None):
        super().__init__()

        self.render_mode  = render_mode
        self.ctrl_hz      = ctrl_hz        # policy frequency
        self.sim_hz       = sim_hz
        self.ctrl_decim   = sim_hz // ctrl_hz
        self.dt_ctrl      = 1.0 / ctrl_hz
        self.max_steps    = int(10.0 * ctrl_hz)  # 10 second episodes

        # shared step counter for curriculum (set externally by callback)
        self._total_steps = 0 if total_steps_ref is None else total_steps_ref

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

        # Sample terrain from curriculum
        cfg = self._get_curriculum()
        terrain_probs = [cfg["flat"], cfg["slope"], cfg["step"]]
        # normalize
        tp = np.array(terrain_probs)
        tp = tp / tp.sum()
        choice = rng.choice(["flat", "slope", "step"], p=tp)

        slope_deg   = 0.0
        step_height = 0.0
        xml_to_load = str(self.xml_path)

        if choice == "slope":
            slope_deg = rng.uniform(-cfg["slope_max"], cfg["slope_max"])
            # avoid tiny slopes
            if abs(slope_deg) < 2.0:
                slope_deg = np.sign(slope_deg) * 2.0 if slope_deg != 0 else 5.0
            direction = "slope"
            xml_to_load = self._make_terrain_xml("slope", slope_deg=slope_deg)
            self._terrain_type = f"slope_{slope_deg:.1f}deg"

        elif choice == "step":
            step_height = rng.uniform(0.03, cfg["step_max"])
            direction   = rng.choice(["step_up", "step_down"])
            xml_to_load = self._make_terrain_xml(direction, step_height=step_height)
            self._terrain_type = f"{direction}_{step_height*100:.1f}cm"

        else:
            self._terrain_type = "flat"

        # Load model for this terrain
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_to_load)
        except Exception:
            # Fallback to flat if XML generation failed
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self._terrain_type = "flat"

        # Clean up temp file
        if xml_to_load != str(self.xml_path) and os.path.exists(xml_to_load):
            try:
                os.unlink(xml_to_load)
            except Exception:
                pass

        self.model.opt.timestep = 1.0 / self.sim_hz
        self.data = mujoco.MjData(self.model)

        # Initialize robot pose with small noise
        self.data.qpos[:3]  = [0.0, 0.0, 0.35]          # base position
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w first for MuJoCo
        self.data.qpos[7:]  = Q_STAND + rng.uniform(-0.05, 0.05, 12)
        self.data.qvel[:]   = rng.uniform(-0.05, 0.05, self.model.nv)

        mujoco.mj_forward(self.model, self.data)

        self._prev_action  = np.zeros(self.n_joints)
        self._step_count   = 0
        self._phase        = rng.uniform(0, 2 * np.pi)   # random phase start
        self._episode_reward = 0.0
        self._init_yaw     = 0.0   # base starts at identity quaternion -> yaw = 0

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

        # Advance gait phase
        self._phase = (self._phase + 2 * np.pi * GAIT_HZ * self.dt_ctrl) % (2 * np.pi)
        self._step_count  += 1
        self._total_steps += 1

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

        # Gait phase
        phase_sin = np.sin(self._phase)
        phase_cos = np.cos(self._phase)

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
        r_yaw = -0.5 * yaw_err**2

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

        # Foot contact + clearance (fix: swing feet weren't lifting enough)
        foot_names = ["FL_foot", "FR_foot",
                      "RL_foot", "RR_foot"]
        r_clearance = 0.0
        for fname in foot_names:
            try:
                fid = self.model.body(fname).id
                foot_z = d.xpos[fid, 2]
                in_contact = foot_z < 0.05
                if not in_contact:  # swing foot -- reward lifting
                    r_clearance += np.clip(foot_z, 0, 0.15) * 0.5
            except Exception:
                pass

        # Imitation reward -- match MPC reference joint angles at current phase
        r_imitate = 0.0
        if _REF_Q_FWD is not None:
            # Find closest reference frame by phase
            phase_diffs = np.abs(_REF_PHASE_FWD - self._phase)
            # Handle wraparound
            phase_diffs = np.minimum(phase_diffs, 2*np.pi - phase_diffs)
            ref_idx = int(np.argmin(phase_diffs))
            ref_q   = _REF_Q_FWD[ref_idx]

            # Current joint angles in actuator order
            q_now = d.qpos[7:19][JOINT_REORDER]

            # Exponential imitation reward -- 1.0 when matching perfectly
            joint_err = np.sum((q_now - ref_q)**2)
            r_imitate = np.exp(-2.0 * joint_err)

        return float(r_upright + r_yaw + r_height + r_forward + r_lateral +
                     r_torque + r_smooth + r_clearance + r_imitate)

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
