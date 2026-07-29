"""
Behavior-cloning pretraining: instead of letting PPO discover the gait purely
through trial-and-error reward-following, directly regress the policy's mean
action onto what the recorded MPC reference trajectory implies, at every
recorded frame. This trains the SAME PPO model object (same net_arch, same
policy) that train_ppo.py's --resume flag will later fine-tune with RL --
no separate network / weight-transplant needed.

Run on your machine, from the repo root.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, 'examples')
from go2_terrain_env import Go2TerrainEnv, Q_STAND, Q_MIN, Q_MAX, JOINT_REORDER
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ------------------------------------------------------------------
# 1. Load reference data and build (observation, target_action) pairs
# ------------------------------------------------------------------
REF_PATH = "examples/data/mpc_reference.npz"
XML_PATH = "/home/aryaman/go2-convex-mpc/models/MJCF/go2/scene.xml"
REF_DT   = 1.0 / 200.0   # reference was recorded at 200Hz -- confirm this matches
                          # your collection script's actual rate; if it doesn't,
                          # only the finite-difference base velocities below are
                          # affected (everything else is rate-independent).

ref = np.load(REF_PATH)

def build_dataset(gait_prefix):
    q     = ref[f"{gait_prefix}_q"]          # (N, 12) actuator order
    dq    = ref[f"{gait_prefix}_dq"]         # (N, 12) actuator order
    base_pos  = ref[f"{gait_prefix}_base_pos"]   # (N, 3)
    base_quat = ref[f"{gait_prefix}_base_quat"]  # (N, 4) assumed (w,x,y,z)
    mask  = ref[f"{gait_prefix}_mask"]       # (N, 4)
    phase = ref[f"{gait_prefix}_phase"]      # (N,)
    N = q.shape[0]

    # Base linear velocity via central finite difference (base_pos wasn't
    # accompanied by a recorded velocity -- approximate it here).
    lin_vel = np.zeros((N, 3))
    lin_vel[1:-1] = (base_pos[2:] - base_pos[:-2]) / (2 * REF_DT)
    lin_vel[0]  = (base_pos[1] - base_pos[0]) / REF_DT
    lin_vel[-1] = (base_pos[-1] - base_pos[-2]) / REF_DT

    # Base angular velocity via finite-differenced roll/pitch/yaw (approximate
    # -- not a rigorous body-frame angular velocity, but sufficient signal for
    # BC pretraining; PPO fine-tuning afterward corrects any residual mismatch
    # through real environment interaction).
    def quat_to_rpy(qq):
        # Reference quaternions are stored as (x, y, z, w) -- confirmed via
        # diagnostic (identity quat [0,0,0,1] has w last, not first like
        # MuJoCo's own qpos convention).
        qx, qy, qz, qw = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        return roll, pitch, yaw

    roll, pitch, yaw = quat_to_rpy(base_quat)
    ang_vel = np.zeros((N, 3))
    for arr, col in [(roll, 0), (pitch, 1), (yaw, 2)]:
        d = np.zeros(N)
        d[1:-1] = (arr[2:] - arr[:-2]) / (2 * REF_DT)
        d[0]  = (arr[1] - arr[0]) / REF_DT
        d[-1] = (arr[-1] - arr[-2]) / REF_DT
        ang_vel[:, col] = d

    # Gravity vector in body frame, per frame
    g_body = np.zeros((N, 3))
    for i in range(N):
        qx, qy, qz, qw = base_quat[i]   # (x, y, z, w) order -- see note above
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])
        g_body[i] = R.T @ np.array([0, 0, -1.0])

    base_height = base_pos[:, 2:3]

    # Target action: inverse of q_target = Q_STAND + action*0.5, clipped.
    # Q_STAND and Q_MIN/Q_MAX are identical across all 4 leg groups, so no
    # reordering subtlety applies here -- see chat discussion for derivation.
    target_action = np.clip((q - Q_STAND) * 2.0, -1.0, 1.0)

    # prev_action feature: shifted target_action (what the env would show as
    # self._prev_action if this trajectory were actually being executed)
    prev_action = np.zeros_like(target_action)
    prev_action[1:] = target_action[:-1]

    phase_sin = np.sin(phase)
    phase_cos = np.cos(phase)

    obs = np.concatenate([
        roll[:, None], pitch[:, None],   # 2
        ang_vel,                          # 3
        lin_vel,                          # 3
        q,                                 # 12
        dq,                                # 12
        prev_action,                       # 12
        phase_sin[:, None], phase_cos[:, None],  # 2
        g_body,                            # 3
        mask,                              # 4
        base_height,                       # 1
    ], axis=1).astype(np.float32)

    return obs, target_action.astype(np.float32)

obs_fwd, act_fwd = build_dataset("trot_forward")
obs_ip,  act_ip  = build_dataset("trot_inplace")

all_obs = np.concatenate([obs_fwd, obs_ip], axis=0)
all_act = np.concatenate([act_fwd, act_ip], axis=0)

print(f"BC dataset (pre-mirror): {all_obs.shape[0]} samples, "
      f"obs_dim={all_obs.shape[1]}, act_dim={all_act.shape[1]}")

# ------------------------------------------------------------------
# 1b. Mirror-augment the dataset (left-right symmetry augmentation).
# Motivation: across many training runs, one diagonal leg pair (FR+RL)
# reliably converges to a clean trot while the other (FL+RR) never does,
# despite a reward that's symmetric by construction. Diagnosis: an
# unconstrained MLP has no guarantee of learning a symmetric solution --
# once one diagonal becomes slightly more reliable it generates cleaner
# gradients, and nothing forces the other diagonal to catch up. Mirroring
# every training example (so the network sees the SAME amount of "FL+RR
# swinging" data as "FR+RL swinging" data, from literally reflected copies
# of the same trajectories) directly addresses this rather than hoping
# gradient descent discovers the symmetry unaided.
#
# Mirror convention (block order in q/dq/action/prev_action is actuator
# order FR, FL, RR, RL; mask order is FL, FR, RL, RR -- both use the same
# block permutation [1,0,3,2] since it's just "swap first pair, swap second
# pair" regardless of which leg is listed first within each pair):
#   - hip joint (index 0 of each 3-block): SIGN FLIPS under mirroring --
#     verified numerically via forward kinematics (RMS error 0.000000 with
#     flip vs 0.0566 without, see diag_mirror_verify.py), not assumed from
#     the FL/FR labels.
#   - thigh, calf (indices 1,2 of each block): unchanged.
#   - roll -> -roll, pitch unchanged (reflection across sagittal plane)
#   - angular velocity is a pseudovector: [wx,wy,wz] -> [-wx, wy, -wz]
#   - linear velocity / gravity are polar vectors: [x,y,z] -> [x,-y,z]
#   - phase_sin/cos both negate (mirroring shifts phase by pi)
#   - base_height unchanged
BLOCK_PERM = [1, 0, 3, 2]  # swap FR<->FL, RR<->RL (or FL<->FR, RL<->RR -- same permutation)

def mirror_leg_blocks(arr, flip_hip):
    """arr: (N, 12), 4 blocks of 3 (hip, thigh, calf). Permutes blocks per
    BLOCK_PERM; optionally flips the sign of the hip (index 0) component."""
    out = np.zeros_like(arr)
    for orig_block in range(4):
        new_block = BLOCK_PERM[orig_block]
        src = arr[:, orig_block*3:(orig_block+1)*3]
        out[:, new_block*3]     = (-1 if flip_hip else 1) * src[:, 0]
        out[:, new_block*3 + 1] = src[:, 1]
        out[:, new_block*3 + 2] = src[:, 2]
    return out

def mirror_action(act):
    return mirror_leg_blocks(act, flip_hip=True)

def mirror_obs(obs):
    out = obs.copy()
    out[:, 0]      = -obs[:, 0]                       # roll
    # pitch (index 1) unchanged
    out[:, 2]      = -obs[:, 2]                       # ang_vel x
    out[:, 4]      = -obs[:, 4]                       # ang_vel z (index 3 = y unchanged)
    out[:, 6]      = -obs[:, 6]                       # lin_vel y (index5=x,7=z unchanged)
    out[:, 8:20]   = mirror_leg_blocks(obs[:, 8:20],  flip_hip=True)   # q
    out[:, 20:32]  = mirror_leg_blocks(obs[:, 20:32], flip_hip=True)   # dq
    out[:, 32:44]  = mirror_leg_blocks(obs[:, 32:44], flip_hip=True)   # prev_action
    out[:, 44]     = -obs[:, 44]                      # phase_sin
    out[:, 45]     = -obs[:, 45]                      # phase_cos
    out[:, 47]     = -obs[:, 47]                      # g_body y (46=x,48=z unchanged)
    mask = obs[:, 49:53]
    out[:, 49:53]  = mask[:, [1, 0, 3, 2]]            # mask: FL,FR,RL,RR -> swap pairs
    # base_height (index 53) unchanged
    return out

mirrored_obs = mirror_obs(all_obs)
mirrored_act = mirror_action(all_act)

all_obs = np.concatenate([all_obs, mirrored_obs], axis=0)
all_act = np.concatenate([all_act, mirrored_act], axis=0)

print(f"BC dataset (mirror-augmented): {all_obs.shape[0]} samples "
      f"({all_obs.shape[0]//2} original + {all_obs.shape[0]//2} mirrored)")

# ------------------------------------------------------------------
# 2. Instantiate the SAME PPO model architecture used for RL fine-tuning
# ------------------------------------------------------------------
gte_env = Go2TerrainEnv(xml_path=XML_PATH)
vec_env = DummyVecEnv([lambda: gte_env])

model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                        activation_fn=torch.nn.ELU),
    device="auto", verbose=0,
)

assert all_obs.shape[1] == model.observation_space.shape[0], \
    f"obs dim mismatch: dataset={all_obs.shape[1]} vs env={model.observation_space.shape[0]}"

# ------------------------------------------------------------------
# 3. Supervised regression directly on model.policy's own parameters
# ------------------------------------------------------------------
device = model.policy.device
obs_t = torch.tensor(all_obs, device=device)
act_t = torch.tensor(all_act, device=device)

n = obs_t.shape[0]
n_val = max(1, int(0.1 * n))
perm = torch.randperm(n)
val_idx, train_idx = perm[:n_val], perm[n_val:]

optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
BATCH_SIZE = 256
N_EPOCHS = 200

for epoch in range(N_EPOCHS):
    model.policy.train()
    perm_train = train_idx[torch.randperm(train_idx.shape[0])]
    total_loss = 0.0
    for i in range(0, perm_train.shape[0], BATCH_SIZE):
        batch_idx = perm_train[i:i+BATCH_SIZE]
        obs_batch = obs_t[batch_idx]
        act_batch = act_t[batch_idx]

        dist = model.policy.get_distribution(obs_batch)
        pred_mean = dist.distribution.mean

        loss = F.mse_loss(pred_mean, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * obs_batch.shape[0]

    if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
        model.policy.eval()
        with torch.no_grad():
            val_dist = model.policy.get_distribution(obs_t[val_idx])
            val_loss = F.mse_loss(val_dist.distribution.mean, act_t[val_idx]).item()
        print(f"epoch {epoch:4d}  train_mse={total_loss/train_idx.shape[0]:.5f}  val_mse={val_loss:.5f}")

# ------------------------------------------------------------------
# 4. Save -- compatible with your existing train_ppo.py --resume flag
# ------------------------------------------------------------------
model.save("go2_imitate_bc_pretrained")
print("Saved go2_imitate_bc_pretrained.zip")
