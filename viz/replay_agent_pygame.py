import math
import time
import pygame
import numpy as np
from stable_baselines3 import PPO
from envs.racing_env import RacingEnv

BG = (20,24,28)
FG = (230,230,230)

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

def main(model_path="data/models/ppo_racing_env_v1.zip", deterministic=False):
    # Env bez grafiky (stav držíme v envu)
    env = RacingEnv()
    # Pygame okno podle rozměrů envu
    pygame.init()
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Racing-AI | PPO Agent Replay")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Načti model
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Nemohu načíst model {model_path}: {e}")
        return

    # Reset epizody
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    paused = False

    # FPS podle env.dt (typicky 60)
    target_fps = int(round(1.0 / env.dt))

    running = True
    while running:
        dt_ms = clock.tick(target_fps)
        # Eventy
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
                    deterministic = not deterministic  # přepínač determinismu
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, f"screenshot_{int(time.time())}.png")

       # Logika agenta
        if not paused:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.array(action, dtype=float)

            # AUTO-START: pokud stojíme nebo jsme na začátku, vnuti dopředný plyn
            if steps < 180 or abs(env.vel) < 5.0:   # ~3 s při 60 FPS
                action[1] = max(action[1], 0.6)     # min. plyn 0.6

            obs, r, done, trunc, info = env.step(action)
            total_r += r
            steps += 1

            # auto-reset po konci epizody
            if done or trunc:
                obs, _ = env.reset()
                total_r = 0.0
                steps = 0
                # malý "odpich" i po resetu
                action = np.array([0.0, 0.7], dtype=float)
                obs, _, _, _, _ = env.step(action)

        # Render
        screen.fill(BG)
        draw_track(screen, env.centerline, env.TRACK_WIDTH)
        draw_car(screen, env.x, env.y, env.heading)

        hud_lines = [
            f"Steps: {steps}   Reward (episode): {total_r:8.3f}",
            f"Speed: {env.vel:7.1f} px/s   Deterministic: {deterministic}",
            f"Action: steer_rate={float(action[0]) if not paused else 0:+.2f}  throttle={float(action[1]) if not paused else 0:+.2f}",
            f"off_track: {getattr(env, 'last_off', False)}   ds: {getattr(env, 'last_ds', 0):+.3f}",
            "Keys: SPACE pause | R reset | D deterministic | S screenshot | ESC quit",
        ]
        hud(screen, font, hud_lines)

        pygame.display.flip()

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()