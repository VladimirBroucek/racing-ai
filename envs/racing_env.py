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

def build_rect_centerline(width: int, height: int, margin: int = 140, points_per_edge: int = 40):
    """
    Build a rectangular centerline polyline inside the window.

    width, height : total env/screen size
    margin        : distance from window border to the rectangle
    points_per_edge: how many points to use per edge (controls smoothness of motion)
    """
    left = margin
    right = width - margin
    top = margin
    bottom = height - margin

    pts = []

    # Top edge: left -> right
    for i in range(points_per_edge):
        t = i / (points_per_edge - 1)
        x = left + t * (right - left)
        y = top
        pts.append((x, y))

    # Right edge: top -> bottom
    for i in range(points_per_edge):
        t = i / (points_per_edge - 1)
        x = right
        y = top + t * (bottom - top)
        pts.append((x, y))

    # Bottom edge: right -> left
    for i in range(points_per_edge):
        t = i / (points_per_edge - 1)
        x = right - t * (right - left)
        y = bottom
        pts.append((x, y))

    # Left edge: bottom -> top
    for i in range(points_per_edge):
        t = i / (points_per_edge - 1)
        x = left
        y = bottom - t * (bottom - top)
        pts.append((x, y))

    # Close loop explicitly by repeating the first point
    if pts[0] != pts[-1]:
        pts.append(pts[0])

    return pts

class RacingEnv(gym.Env):

    metadata = {"render_modes": ["ansi"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()
       
        self.W, self.H = 1000, 700
       
        self.TRACK_WIDTH = 60.0

        # Build a rectangular track centerline with sharp corners
        self.centerline = build_rect_centerline(self.W, self.H, margin=140, points_per_edge=40)
        self.centerline_lengths = polyline_prefix_lengths(self.centerline)
        self.track_length = self.centerline_lengths[-1]

        self.last_ds = 0.0
        self.last_bearing_err = 0.0

        self.grace_steps = 30   # ~0.5 s při 60 FPS, můžeš dát 60-120 pro 1–2 s

        self.laps = 0
        self.lap_s = 0.0

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
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),  # 6 features: lat, heading, speed, progress_dir, bearing, dist
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.max_steps = 4000   # ~ 4000 kroků po 1/60 s ≈ 66 s simulace
        self.dt = 1.0/60.0

        self.reset(seed=seed)

    def _locate_on_track(self, px: float, py: float):
        """
        Find nearest point on the centerline to (px, py).
        Returns:
            seg_i  : index of segment [i, i+1]
            seg_t  : local parameter in [0, 1] along that segment
            lat    : signed lateral offset from centerline (pixels)
            h_err  : heading error between car heading and segment direction (radians)
        """
        best_d2 = 1e18
        best_i = 0
        best_t = 0.0
        best_lat = 0.0
        best_h_err = 0.0

        n = len(self.centerline)
        for i in range(n - 1):
            ax, ay = self.centerline[i]
            bx, by = self.centerline[i + 1]

            abx, aby = (bx - ax), (by - ay)
            ab2 = abx * abx + aby * aby
            if ab2 == 0.0:
                continue

            apx, apy = (px - ax), (py - ay)
            t = (apx * abx + apy * aby) / ab2
            t_clamped = max(0.0, min(1.0, t))

            qx = ax + abx * t_clamped
            qy = ay + aby * t_clamped

            dx, dy = px - qx, py - qy
            d2 = dx * dx + dy * dy

            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_t = t_clamped

                # Lateral offset: projection of (dx, dy) on left normal
                seg_len = math.sqrt(ab2) + 1e-9
                nx, ny = abx / seg_len, aby / seg_len  # tangent
                lx, ly = -ny, nx                        # left normal
                lat = dx * lx + dy * ly

                # Heading error: difference between car heading and segment direction
                seg_heading = math.atan2(aby, abx)
                h_err = (self.heading - seg_heading + math.pi) % (2 * math.pi) - math.pi

                best_lat = lat
                best_h_err = h_err

        return best_i, best_t, best_lat, best_h_err

    def _teczna_normala(self, seg_i):
        ax, ay = self.centerline[seg_i]
        bx, by = self.centerline[seg_i+1]
        tx, ty = (bx-ax), (by-ay)
        L = math.hypot(tx, ty) + 1e-9
        nx, ny = tx/L, ty/L
        lx, ly = -ny, nx
        return nx, ny, lx, ly, L
    
    def _obs(self):

        """
        Returns normalized observation including a lookahead target.
        """
        # Locate car on the track centerline
        seg_i, seg_t, lat, h_err = self._locate_on_track(self.x, self.y)
        # Removed the line that overwrites previous segment indices:
        # self.prev_seg_i, self.prev_seg_t = seg_i, seg_t

        # Base normalized features
        lat_norm = np.clip(lat / (self.TRACK_WIDTH * 0.5), -1.0, 1.0)
        heading_norm = math.sin(h_err)  # compact and bounded encoding
        speed_norm = np.clip(self.vel / self.VEL_MAX_FWD, -1.0, 1.0)
        progress_dir = np.sign(self.last_ds if hasattr(self, "last_ds") else 1.0)

        # Lookahead target along centerline
        LOOKAHEAD_DIST = 120.0  # pixels ahead along the centerline
        s_now = self._arc_length(seg_i, seg_t)
        s_target = (s_now + LOOKAHEAD_DIST) % self.track_length
        tx, ty = self._pos_on_centerline(s_target)

        # Vector and bearing to target
        dx, dy = (tx - self.x), (ty - self.y)
        dist = math.hypot(dx, dy) + 1e-9
        bearing = math.atan2(dy, dx)
        # Smallest signed angle difference between car heading and target direction
        bearing_err = (bearing - self.heading + math.pi) % (2 * math.pi) - math.pi

        # Normalized lookahead features
        bearing_norm = np.clip(bearing_err / math.pi, -1.0, 1.0)
        dist_norm = np.clip(dist / LOOKAHEAD_DIST, 0.0, 1.0)

        # Store bearing_err for reward shaping in step()
        self.last_bearing_err = bearing_err

        obs = np.array([
            lat_norm,          # lateral deviation (−1: left edge, 0: center, +1: right edge)
            heading_norm,      # compact heading error encoding (sin)
            speed_norm,        # normalized speed
            progress_dir,      # +1 forward, −1 backward
            bearing_norm,      # angle to lookahead target (normalized)
            dist_norm          # distance ratio to lookahead target (0..1)
        ], dtype=np.float32)

        return obs, (seg_i, seg_t, lat, h_err)
    
    def _progress_delta(self, prev_seg_i, prev_seg_t, seg_i, seg_t):
        """Compute change in arc-length along the track (positive = forward)."""
        s_prev = self._arc_length(prev_seg_i, prev_seg_t)
        s_now = self._arc_length(seg_i, seg_t)
        ds = s_now - s_prev
        # Handle wrap-around across the start/finish line
        if ds < -self.track_length * 0.5:
            ds += self.track_length
        elif ds > self.track_length * 0.5:
            ds -= self.track_length
        return ds
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # start poblíž spodní části oválu
        start_i = 0
        ax, ay = self.centerline[start_i]
        bx, by = self.centerline[start_i + 1]
        tx, ty = (bx - ax), (by - ay)
        L = math.hypot(tx, ty) + 1e-9
        nx, ny = tx / L, ty / L      # jednotkový tečný vektor
        lx, ly = -ny, nx             # levá normála

        # start přesně na centerline (můžeš přidat drobný posun dovnitř trati např. -2 px po normále)
        self.x, self.y = ax, ay
        self.heading = math.atan2(ny, nx)   # heading zarovnaný s tečnou
        # volitelně: jemný posun směrem ke středu trati:
        # self.x += lx * 0.0
        # self.y += ly * 0.0

        self.vel = 0.0
        self.steps = 0
        self.time = 0.0
        self.laps = 0
        self.lap_s = 0.0

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
        ds = self._progress_delta(self.prev_seg_i, self.prev_seg_t, seg_i, seg_t)
        self.last_ds = ds

        if ds > 0:
            self.lap_s += ds
            while self.lap_s >= self.track_length:
                self.lap_s -= self.track_length
                self.laps += 1

        # Update previous segment position for the next step
        self.prev_seg_i, self.prev_seg_t = seg_i, seg_t

        # --- wall / track boundary logic ---
        half_w = self.TRACK_WIDTH * 0.5
        lat_ratio = abs(lat) / max(half_w, 1e-6)  # 0 = center, 1 = at wall
        # Hard off-track when crossing the virtual wall
        off_track = lat_ratio > 1.0
        # Soft "near wall" zone to discourage cutting corners but keep episode alive
        near_wall = (lat_ratio > 0.85) and not off_track

        # During initial grace period, never terminate for off-track
        if self.steps < self.grace_steps:
            off_track = False

        # --- reward shaping ---
        # Positive for forward progress, negative for large lateral error, steering abuse and hugging walls
        r = 0.0

        # Progress term (scaled down so values stay reasonable)
        r += 0.01 * (ds / max(1e-6, self.dt))

        # Penalize lateral error: small near center, strong near edges (quadratic)
        r -= 0.30 * (lat_ratio ** 2)

        # Extra penalty for being very close to a wall (but not yet off-track)
        if near_wall:
            r -= 1.0

        # Penalize large steering input (reduce twitchy driving)
        r -= 0.02 * abs(steer_in)

        # Small constant time pressure
        r -= 0.001

        # Encourage pointing towards the lookahead target
        bearing_err = getattr(self, "last_bearing_err", 0.0)
        r += 0.10 * math.cos(bearing_err)

        terminated = False
        truncated = False

        if off_track:
            r -= 5.0
            terminated = True

        self.steps += 1
        self.time += self.dt
        if self.steps >= self.max_steps:
            truncated = True

        info = {
            "off_track": off_track,
            "near_wall": near_wall,
            "ds": ds,
            "laps": self.laps,
            "lap_progress": self.lap_s / self.track_length,
        }

        # For replay/debug HUD
        self.last_off = off_track
        self.last_near_wall = near_wall
        self.last_lat_ratio = lat_ratio
        self.last_ds = ds
        self.last_laps = self.laps
        self.last_lap_progress = self.lap_s / self.track_length

        return obs, r, terminated, truncated, info
    
    def render(self):
        return "RacingEnv(no-graphics)"
    
    def close(self):
        pass

    def _arc_length(self, seg_i, seg_t):
        """Return arc-length along the centerline for (segment index, local t)."""
        return self.centerline_lengths[seg_i] + seg_t * (
            self.centerline_lengths[seg_i + 1] - self.centerline_lengths[seg_i]
        )
    
    def _pos_on_centerline(self, s):
        """Return (x, y) on the centerline for a given arc-length s."""
        i = np.searchsorted(self.centerline_lengths, s, side="right") - 1
        i = int(np.clip(i, 0, len(self.centerline) - 2))
        denom = (self.centerline_lengths[i + 1] - self.centerline_lengths[i] + 1e-9)
        t = (s - self.centerline_lengths[i]) / denom
        ax, ay = self.centerline[i]
        bx, by = self.centerline[i + 1]
        return ax + (bx - ax) * t, ay + (by - ay) * t


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