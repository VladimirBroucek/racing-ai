import os
import time
import math
import glob
import pygame
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from stable_baselines3 import PPO
from envs.racing_env import RacingEnv

# ---------- Nastavení ----------
CHECKPOINT_DIR = "data/models/ckpts"        # kam CheckpointCallback ukládá
FALLBACK_MODEL = "data/models/ppo_racing_env_v1.zip"   # když zatím nejsou checkpointy
CHECK_INTERVAL_SEC = 2.0                    # jak často kontrolovat nové checkpointy
BG = (20, 24, 28)
FG = (230, 230, 230)

# ---------- Geometrie & kreslení ----------
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

def draw_car(surface, x, y, heading, color=(0, 200, 255)):
    L = 34
    Wc = 18
    pts_local = [( L/2, 0), (-L/2, -Wc/2), (-L/2,  Wc/2)]
    c, s = math.cos(heading), math.sin(heading)
    pts_world = []
    for px, py in pts_local:
        wx = x + px * c - py * s
        wy = y + px * s + py * c
        pts_world.append((wx, wy))
    pygame.draw.polygon(surface, color, pts_world)
    nose_x = x + math.cos(heading) * (L/2 + 8)
    nose_y = y + math.sin(heading) * (L/2 + 8)
    pygame.draw.line(surface, (255, 255, 255), (x, y), (nose_x, nose_y), 2)

def draw_track(surface, centerline, track_width):
    pygame.draw.lines(surface, (70, 90, 110), False, centerline, 2)
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

# ---------- Watcher ----------
def latest_checkpoint(path: str) -> Optional[Tuple[str, float]]:
    """Vrátí (soubor, mtime) pro nejnovější .zip v cestě, nebo None."""
    if not os.path.isdir(path):
        return None
    files = sorted(glob.glob(os.path.join(path, "*.zip")), key=os.path.getmtime)
    if not files:
        return None
    f = files[-1]
    return f, os.path.getmtime(f)

def load_model(model_path: str, env: RacingEnv) -> Optional[PPO]:
    try:
        model = PPO.load(model_path, env=env)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Cannot load model {model_path}: {e}")
        return None

def soft_recover(env: RacingEnv):
    """Vrátí auto na nejbližší centerline, srovná heading a zastaví."""
    qx, qy, seg_i, seg_t = nearest_point_on_polyline(env.x, env.y, env.centerline)
    ax, ay = env.centerline[seg_i]; bx, by = env.centerline[seg_i+1]
    tx, ty = (bx-ax), (by-ay); L = (tx*tx + ty*ty) ** 0.5 + 1e-9
    nx, ny = tx/L, ty/L
    env.x, env.y = qx, qy
    env.heading = math.atan2(ny, nx)
    env.vel = 0.0

def main():
    # Env
    env = RacingEnv()
    pygame.init()
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Racing-AI | Watcher")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Najdi startovní model
    ck = latest_checkpoint(CHECKPOINT_DIR)
    current_path = ck[0] if ck else (FALLBACK_MODEL if os.path.exists(FALLBACK_MODEL) else None)
    current_mtime = ck[1] if ck else (os.path.getmtime(FALLBACK_MODEL) if current_path else 0.0)
    if not current_path:
        print("[WARN] No model available. Train first or set FALLBACK_MODEL.")
        return

    model = load_model(current_path, env)
    obs, _ = env.reset()

    # Řídicí stav
    deterministic = True
    force_random = False
    auto_reload = True
    total_r = 0.0
    steps = 0
    last_reload = datetime.now()
    last_check = time.time()
    last_pos = None
    stuck_frames = 0

    # Warmup parametry
    WARMUP_STEPS = 120
    MIN_MOVE_SPEED = 5.0
    STUCK_FRAMES_LIMIT = 60

    running = True
    while running:
        dt_ms = clock.tick(int(round(1.0/env.dt)))
        now = time.time()

        # Auto-reload: kontrola nového checkpointu
        if auto_reload and (now - last_check) >= CHECK_INTERVAL_SEC:
            last_check = now
            ck = latest_checkpoint(CHECKPOINT_DIR)
            if ck and ck[1] > current_mtime:
                current_path, current_mtime = ck
                new_model = load_model(current_path, env)
                if new_model is not None:
                    model = new_model
                    last_reload = datetime.now()
                    # volitelně: reset epizody po reloadu
                    obs, _ = env.reset()
                    total_r = 0.0
                    steps = 0
                    last_pos = None
                    stuck_frames = 0

        # Eventy
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # pauza toggle – zde nemáme, držíme simple (můžeš si přidat)
                    pass
                elif event.key == pygame.K_d:
                    deterministic = not deterministic
                elif event.key == pygame.K_f:
                    force_random = not force_random
                elif event.key == pygame.K_a:
                    auto_reload = not auto_reload
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_r = 0.0
                    steps = 0
                    last_pos = None
                    stuck_frames = 0
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, f"screenshot_{int(time.time())}.png")

        # Akce agenta
        if model is None:
            break

        if force_random:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        action = np.array(action, dtype=float)

        # Warmup & anti-stuck
        if steps < WARMUP_STEPS or abs(env.vel) < MIN_MOVE_SPEED:
            action[1] = max(action[1], 0.7)

        # krok envu
        obs, r, done, trunc, info = env.step(action)
        total_r += r
        steps += 1

        # Soft recovery mimo trať
        if info.get("off_track", False):
            soft_recover(env)
            # nepřerušuj epizodu, prostě pokračuj
            # můžeme ještě jednou obs refreshnout
            obs, _ = env.reset()  # volitelné: reset pro čistý stav metrik
            total_r = 0.0
            steps = 0
            last_pos = None
            stuck_frames = 0

        # Render
        screen.fill(BG)
        draw_track(screen, env.centerline, env.TRACK_WIDTH)
        draw_car(screen, env.x, env.y, env.heading)

        # HUD
        model_name = os.path.basename(current_path)
        hud_lines = [
            f"Model: {model_name}   Reload: {last_reload.strftime('%H:%M:%S')}   AutoReload: {auto_reload}",
            f"Steps: {steps}   Reward(ep): {total_r:8.3f}   FPS: {clock.get_fps():4.1f}",
            f"Speed: {env.vel:7.1f} px/s   Deterministic: {deterministic}   Random: {force_random}",
            f"Laps: {getattr(env,'last_laps',0)}   Lap%: {getattr(env,'last_lap_progress',0.0)*100:5.1f}%",
            f"off_track: {getattr(env,'last_off', False)}   ds: {getattr(env,'last_ds', 0):+.3f}",
            "Keys: D deterministic | F random | A auto-reload | R reset | S screenshot | ESC quit",
        ]
        hud(screen, font, hud_lines)
        pygame.display.set_caption(
            f"Racing-AI | Watcher | {model_name} | FPS {clock.get_fps():.1f} | Speed {env.vel:5.1f}"
        )
        pygame.display.flip()

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()