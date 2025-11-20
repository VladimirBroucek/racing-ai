import argparse
import numpy as np
from stable_baselines3 import PPO
from envs.racing_env import RacingEnv

def evaluate(model_path: str, episodes: int = 10, deterministic: bool = True):
    env = RacingEnv()
    model = PPO.load(model_path, env=env)

    stats = {
        "ep_return": [],
        "steps": [],
        "off_steps": [],
        "progress_sum": [],     
        "speed_sum": [],
        "lat_abs_sum": [],
        "progdir_sum": [],
    }

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        ep_ret = 0.0
        ep_steps = 0
        off_cnt = 0
        prog_sum = 0.0
        speed_sum = 0.0
        lat_abs_sum = 0.0
        progdir_sum = 0.0

        # Small safety so we don't evaluate a completely "dead" policy at the very beginning
        warmup = 5

        while not (done or trunc):
            if warmup > 0:
                action = env.action_space.sample()
                action[0] = 0.0
                action[1] = 0.7
                warmup -= 1
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, r, done, trunc, info = env.step(action)
            ep_ret += r
            ep_steps += 1
            prog_sum += info.get("ds", 0.0)
            speed_sum += abs(env.vel)
            
            lat_norm, h_norm, v_norm, progdir, bearing_norm, dist_norm = obs
            lat_abs_sum += abs(lat_norm)      
            progdir_sum += float(progdir)
            off_cnt += int(getattr(env, "last_off", False))

            if ep_steps > 5000:  
                trunc = True

        stats["ep_return"].append(ep_ret)
        stats["steps"].append(ep_steps)
        stats["off_steps"].append(off_cnt)
        stats["progress_sum"].append(prog_sum)
        stats["speed_sum"].append(speed_sum / max(1, ep_steps))
        stats["lat_abs_sum"].append(lat_abs_sum / max(1, ep_steps))
        stats["progdir_sum"].append(progdir_sum / max(1, ep_steps))

    
    mean_ret = np.mean(stats["ep_return"])
    mean_steps = np.mean(stats["steps"])
    off_rate = (np.sum(stats["off_steps"]) / np.sum(stats["steps"])) if np.sum(stats["steps"]) > 0 else 1.0
    total_prog = np.sum(stats["progress_sum"])
    laps_est = total_prog / env.track_length
    mean_speed = np.mean(stats["speed_sum"])
    mean_lat_abs = np.mean(stats["lat_abs_sum"])
    mean_progdir = np.mean(stats["progdir_sum"])

    print("\n=== EVALUATION ===")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}  |  Deterministic: {deterministic}")
    print(f"Avg return / episode: {mean_ret:8.3f}")
    print(f"Avg steps / episode : {mean_steps:8.1f}")
    print(f"Off-track rate      : {off_rate*100:6.2f}% of steps")
    print(f"Total progress      : {total_prog:8.1f} px  (~ {laps_est:.2f} laps)")
    print(f"Mean speed          : {mean_speed:8.1f} px/s")
    print(f"Mean |lat_norm|     : {mean_lat_abs:8.3f}  (0 = center, 1 = track edge)")
    print(f"Mean progress_dir   : {mean_progdir:8.3f}  (-1 = backward, +1 = forward)\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/ppo_rect_v1.zip")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--stochastic", action="store_true", help="Use stochastic policy (deterministic=False)")
    args = p.parse_args()
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        deterministic=not args.stochastic,
    )