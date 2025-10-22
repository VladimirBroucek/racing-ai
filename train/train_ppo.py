import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.racing_env import RacingEnv

def make_env():
    return RacingEnv()

def main():
    env = DummyVecEnv([make_env])  # 1 paralelní env na start

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        learning_rate=3e-4,
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,          # ← tady byl překlep (vt_oef -> vf_coef)
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="data/tb"
    )

    total = 400_000
    model.learn(total_timesteps=total)

    os.makedirs("data/models", exist_ok=True)
    model.save("data/models/ppo_racing_env_v1")
    print("Model saved to data/models/ppo_racing_env_v1.zip")

if __name__ == "__main__":
    main()