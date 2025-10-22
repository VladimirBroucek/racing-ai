import time
import numpy as np
from stable_baselines3 import PPO
from envs.racing_env import RacingEnv

def run_episode(model_path="data/models/ppo_racing_env_v1.zip", episodes=3):
    env = RacingEnv()
    model = PPO.load(model_path, env=env)

    for ep in range(1, episodes+1):
        obs, _ = env.reset()
        done = False
        trunc = False
        total_r = 0.0
        steps = 0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            total_r += r
            steps += 1
            # malá pauza ať to „nežere“ CPU
            time.sleep(env.dt * 0.25)
        print(f"Episode {ep}: return={total_r:.2f}, steps={steps}, off_track={info.get('off_track', False)}")

if __name__ == "__main__":
    run_episode()        