import math
import time
import pygame
import numpy as np
from stable_baselines3 import PPO
from envs.racing_env import RacingEnv

BG = (20,24,28)
FG = (230,230,230)

def project_point_to_segment(px, py, ax, ay, bx, by):
    abx, aby = (bx - ax), (by - ay)
    ab2 = abx*abx + aby*aby
    if ab2 == 0:
        return ax, ay, 0.0
    apx, apy = (px - ax), (py - ay)
    t = max(0.0, min(1.0, (apx*abx + apy*aby) / ab2))
    qx, qy = ax + t*abx, ay + t*aby
    return qx, qy, t

def nearest_point_on_polyline(px, py, pts):
    best = (pts[0][0], pts[0][1]); best_d2 = 1e18; best_i = 0; best_t = 0.0
    for i in range(len(pts)-1):
        ax, ay = pts[i]; bx, by = pts[i+1]
        qx, qy, t = project_point_to_segment(px, py, ax, ay, bx, by)
        d2 = (px - qx)**2 + (py - qy)**2
        if d2 < best_d2:
            best_d2 = d2; best = (qx, qy); best_i = i; best_t = t
    return best[0], best[1], best_i, best_t

def draw_car(surface, x, y, heading, color=(0,200,255)):
    L = 34
    Wc = 18
    pts_local = [
        ( L/2, 0),
        (-L/2, -Wc/2),
        (-L/2,  Wc/2),
    ]
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    pts_world = []
    for px, py in pts_local:
        wx = x + px * cos_h - py * sin_h
        wy = y + px * sin_h + py * cos_h
        pts_world.append((wx, wy))
    pygame.draw.polygon(surface, color, pts_world)
    nose_x = x + math.cos(heading) * (L/2 + 8)
    nose_y = y + math.sin(heading) * (L/2 + 8)
    pygame.draw.line(surface, (255, 255, 255), (x, y), (nose_x, nose_y), 2)

def draw_track(surface, centerline, track_width):
    # centerline
    pygame.draw.lines(surface, (70, 90, 110), False, centerline, 2)
    # hrany (řidší sampling kvůli výkonu)
    step = 8
    left_pts, right_pts = [], []
    for i in range(0, len(centerline)-1, step):
        ax, ay = centerline[i]
        bx, by = centerline[i+1]
        tx, ty = (bx-ax), (by-ay)
        Lseg = math.hypot(tx, ty) + 1e-9
        nx, ny = tx/Lseg, ty/Lseg
        lx, ly = -ny, nx
        off = track_width * 0.5
        left_pts.append((ax + lx*off, ay + ly*off))
        right_pts.append((ax - lx*off, ay - ly*off))
    if len(left_pts) > 1:
        pygame.draw.lines(surface, (40, 130, 40), False, left_pts, 2)
        pygame.draw.lines(surface, (130, 40, 40), False, right_pts, 2)

def hud(surface, font, lines, x=20, y=20, color=FG, dy=22):
    for i, line in enumerate(lines):
        txt = font.render(line, True, color)
        surface.blit(txt, (x, y + i*dy))

def main(model_path="data/models/ppo_rect_v1.zip", deterministic=False):
    # --- replay safety & diagnostics ---
    WARMUP_STEPS = 240           # ~4s auto-throttle warmup (60 FPS)
    MIN_MOVE_SPEED = 5.0         # px/s considered "moving"
    STUCK_FRAMES_LIMIT = 60      # if agent outputs ~zero for 1s, kick it
    force_random = False         # toggle with 'F' key
    stuck_frames = 0             # counter of consecutive "stuck" frames
    last_pos = None              # track movement for diagnostics

    # Environment without rendering logic (state is kept inside env)
    env = RacingEnv()
    # Pygame window sized according to the environment
    pygame.init()
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Racing-AI | PPO Agent Replay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Load trained PPO model
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Nemohu načíst model {model_path}: {e}")
        return

    # Reset episode
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    paused = False

    # Target FPS based on env.dt (typically 60)
    target_fps = int(round(1.0 / env.dt))

    running = True
    while running:
        dt_ms = clock.tick(target_fps)
        # Input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_r = 0.0
                    steps = 0
                elif event.key == pygame.K_d:
                    deterministic = not deterministic  # toggle deterministic/stochastic inference
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, f"screenshot_{int(time.time())}.png")
                elif event.key == pygame.K_f:
                    force_random = not force_random  # toggle random actions for debug

        # Agent logic and environment stepping
        if not paused:
            # 1) choose action
            if force_random:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            action = np.array(action, dtype=float)

            # 2) Auto-throttle warmup & stuck-kicker
            if steps < WARMUP_STEPS or abs(env.vel) < MIN_MOVE_SPEED:
                action[1] = max(action[1], 0.7)  # ensure forward motion
            # detect "stuck": tiny throttle & near-zero steer and no displacement
            if last_pos is not None:
                dx = env.x - last_pos[0]
                dy = env.y - last_pos[1]
                moved = (dx*dx + dy*dy) ** 0.5 > 0.5
            else:
                moved = True
            zeroish = (abs(action[0]) < 0.05 and abs(action[1]) < 0.05)
            if (not moved and zeroish) or abs(env.vel) < 0.5:
                stuck_frames += 1
            else:
                stuck_frames = 0
            if stuck_frames > STUCK_FRAMES_LIMIT:
                # kick with a small random steer and throttle
                action[0] = float(np.clip(np.random.uniform(-0.3, 0.3), -1, 1))
                action[1] = float(np.clip(np.random.uniform(0.6, 0.9), -1, 1))
                stuck_frames = 0

            # 3) environment step
            obs, r, done, trunc, info = env.step(action)
            total_r += r
            steps += 1
            last_pos = (env.x, env.y)

            # SOFT RECOVERY: when off-track, snap back to centerline without full reset
            if info.get("off_track", False):
                # Project current position to the nearest point on the centerline
                qx, qy, seg_i, seg_t = nearest_point_on_polyline(env.x, env.y, env.centerline)
                # Tangent direction at that segment and heading aligned with it
                ax, ay = env.centerline[seg_i]; bx, by = env.centerline[seg_i+1]
                tx, ty = (bx-ax), (by-ay); L = (tx*tx + ty*ty) ** 0.5 + 1e-9
                nx, ny = tx/L, ty/L
                # Snap car to projected point, align heading and zero out velocity
                env.x, env.y = qx, qy
                env.heading = math.atan2(ny, nx)
                env.vel = 0.0
                # Continue without resetting the episode, just take the next step
                continue

            # Hard episode end only when truncated (e.g., time limit)
            if trunc:
                # Optionally do a soft reset of episode stats on truncation
                obs, _ = env.reset()
                total_r = 0.0
                steps = 0
                last_pos = None
                stuck_frames = 0

        # Render
        screen.fill(BG)
        draw_track(screen, env.centerline, env.TRACK_WIDTH)
        draw_car(screen, env.x, env.y, env.heading)

        hud_lines = [
            f"Steps: {steps}   Reward(episode): {total_r:8.3f}   FPS: {clock.get_fps():4.1f}",
            f"Speed: {env.vel:7.1f} px/s   Deterministic: {deterministic}   Random: {force_random}",
            f"Action: steer={float(action[0]) if not paused else 0:+.2f}  throttle={float(action[1]) if not paused else 0:+.2f}",
            f"Laps: {getattr(env,'last_laps',0)}   Lap%: {getattr(env,'last_lap_progress',0.0)*100:5.1f}%",
            f"off_track: {getattr(env, 'last_off', False)}   ds: {getattr(env, 'last_ds', 0):+.3f}   stuck_frames: {stuck_frames}",
            "Keys: SPACE pause | R reset | D deterministic | F random | S screenshot | ESC quit",
        ]
        hud(screen, font, hud_lines)

        pygame.display.flip()
        pygame.display.set_caption(f"Racing-AI | PPO Agent Replay  |  FPS {clock.get_fps():.1f}  Speed {env.vel:5.1f}")

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()