import math
import gymnasium as gym # pyright: ignore[reportMissingImports]
from gymnasium import spaces # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]

def make_oval_centerline(cx, cy, rx, ry, n=400):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x = cx + rx * math.cos(a)
        y = cy + ry * math.sin(a)
        pts.append((x, y))
    pts.append(pts[0])
    return pts

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
    best_q = (pts[0][0], pts[0][1])
    best_d2 = 1e18
    best_i = 0
    best_t = 0.0
    for i in range(len(pts)-1):
        ax, ay = pts[i]
        bx, by = pts[i+1]
        qx, qy, t = project_point_to_segment(px, py, ax, ay, bx, by)
        d2 = (px - qx)**2 + (py - qy)**2
        if d2 < best_d2:
            best_d2 = d2
            best_q = (qx, qy)
            best_i = i
            best_t = t
    return best_q[0], best_q[1], best_i, best_t, best_d2

def polyline_prefix_lengths(pts):
    L = [0.0]
    acc = 0.0
    for i in range(len(pts)-1):
        ax, ay = pts[i]
        bx, by = pts[i+1]
        acc += math.hypot(bx-ax, by-ay)
        L.append(acc)
    return L

class RacingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()
       
        self.W, self.H = 1000, 700
       
        self.TRACK_WIDTH = 60.0
        self.centerline = make_oval_centerline(self.W*0.5, self.H*0.5, 350, 220, n=400)
        self.centerline_lengths = polyline_prefix_lengths(self.centerline)
        self.track_length = self.centerline_lengths[-1]

        #car physic
        self.ACCEL = 600.0
        self.BRAKE = 900.0
        self.DRAG  = 0.8
        self.ROLL  = 2.0
        self.VEL_MAX_FWD = 900.0
        self.VEL_MAX_BWD = -300.0
        self.STEER_RATE  = 2.8

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([+1.0, +1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([-2.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([+2.0, +1.0, +1.0, +1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.max_steps = 4000   # ~ 4000 kroků po 1/60 s ≈ 66 s simulace
        self.dt = 1.0/60.0

        self.reset(seed=seed)

    def _teczna_normala(self, seg_i):
        ax, ay = self.centerline[seg_i]
        bx, by = self.centerline[seg_i+1]
        tx, ty = (bx-ax), (by-ay)
        L = math.hypot(tx, ty) + 1e-9
        nx, ny = tx/L, ty/L
        lx, ly = -ny, nx
        return nx, ny, lx, ly, L
    
    def _obs(self):

        # Projekce polohy na trať
        qx, qy, seg_i, seg_t, d2 = nearest_point_on_polyline(self.x, self.y, self.centerline)
        nx, ny, lx, ly, seg_len = self._teczna_normala(seg_i)
        # laterální chyba (podepsaná)
        lat = (self.x - qx)*lx + (self.y - qy)*ly
        lat_norm = np.clip(lat / (self.TRACK_WIDTH*0.5), -2.0, 2.0)
        # heading error v rozsahu [-pi, pi]
        h_err = math.atan2(math.sin(self.heading - math.atan2(ny, nx)),
                           math.cos(self.heading - math.atan2(ny, nx)))
        h_norm = np.clip(h_err / math.pi, -1.0, 1.0)
        # rychlost
        v_norm = np.clip(self.vel / self.VEL_MAX_FWD, -1.0, 1.0)
        # progress_dir: jede auto zhruba v tečném směru?
        # kosinus mezi heading vektorem a tečnou
        hx, hy = math.cos(self.heading), math.sin(self.heading)
        prog_dir = np.clip(hx*nx + hy*ny, -1.0, 1.0)

        return np.array([lat_norm, h_norm, v_norm, prog_dir], dtype=np.float32), (seg_i, seg_t, lat, h_err)
    
    def _progress_delta(self, prev_seg_i, prev_seg_t, seg_i, seg_t, seg_len):
        # změna obloukové vzdálenosti po trati (kladná = vpřed)
        s_prev = self.centerline_lengths[prev_seg_i] + prev_seg_t * seg_len
        s_now  = self.centerline_lengths[seg_i] + seg_t * seg_len
        ds = s_now - s_prev
        # ošetři přechod přes start (wrap)
        if ds < -self.track_length * 0.5:
            ds += self.track_length
        elif ds > self.track_length * 0.5:
            ds -= self.track_length
        return ds
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # start poblíž spodní části oválu
        self.x, self.y = self.W*0.5, self.H*0.6
        self.heading = -math.pi/2
        self.vel = 0.0
        self.steps = 0
        self.time = 0.0
        obs, (seg_i, seg_t, lat, h_err) = self._obs()
        self.prev_seg_i, self.prev_seg_t = seg_i, seg_t
        return obs, {}
    
    def step(self, action):
        steer_in = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))

        # zrychlení vpřed/vzad
        acc = self.ACCEL * max(0.0, throttle) + self.BRAKE * min(0.0, throttle)
        # odpory
        drag = self.DRAG * self.vel
        roll = self.ROLL * (1 if self.vel > 0 else -1) if abs(self.vel) > 1e-3 else 0.0
        a_total = acc - drag - roll
        # integrace rychlosti a omezení
        self.vel += a_total * self.dt
        self.vel = min(max(self.vel, self.VEL_MAX_BWD), self.VEL_MAX_FWD)

        # zatáčení škálované rychlostí (ať se na místě neotáčí jak káča)
        speed_factor = max(0.1, min(1.0, abs(self.vel)/400.0))
        self.heading += steer_in * self.STEER_RATE * speed_factor * self.dt

        # posun
        self.x += math.cos(self.heading) * self.vel * self.dt
        self.y += math.sin(self.heading) * self.vel * self.dt

        # metriky
        obs, (seg_i, seg_t, lat, h_err) = self._obs()
        nx, ny, lx, ly, seg_len = self._teczna_normala(seg_i)
        ds = self._progress_delta(self.prev_seg_i, self.prev_seg_t, seg_i, seg_t, seg_len)
        self.prev_seg_i, self.prev_seg_t = seg_i, seg_t

        # off-track?
        off_track = abs(lat) > (self.TRACK_WIDTH * 0.5)

        # --- reward shaping ---
        # kladný za progress, záporný za velkou laterální chybu a prudké řízení
        r = 0.0
        r += 1.0 * (ds / max(1e-6, self.dt)) * 0.01     # progress (škálujeme dolů)
        r -= 0.15 * (abs(lat) / (self.TRACK_WIDTH*0.5)) # drž se uprostřed
        r -= 0.02 * abs(steer_in)                       # netrhej volantem
        r -= 0.001                                      # malý tlak na čas

        terminated = False
        truncated = False

        if off_track:
            r -= 5.0
            terminated = True

        self.steps += 1
        self.time += self.dt
        if self.steps >= self.max_steps:
            truncated = True

        # pro replay/debug (bezpečné i když nenastaveno)
        self.last_off = off_track
        self.last_ds = ds

        info = {"off_track": off_track, "ds": ds}
        return obs, r, terminated, truncated, info
    
    def render(self):
        return "RacingEnv(no-graphics)"
    
    def close(self):
        pass

if __name__ == "__main__":
    env = RacingEnv()
    obs, _ = env.reset()
    ret = 0.0
    for _ in range(300):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if term or trunc:
            break
    print("OK test, return:", ret)
