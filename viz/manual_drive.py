import math
import pygame

pygame.init()
W, H = 1000, 700
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Racing-AI | Manual Drive")
clock = pygame.time.Clock()

BG = (20, 24, 28)
FG = (230,230,230)
font = pygame.font.SysFont("consolas", 18)

running = True

#Car status
x, y = W * 0.5, H * 0.6
heading = -math.pi / 2
vel = 0.0

ACCEL = 600.0
BRAKE = 900.0
DRAG = 0.8
ROLL = 2.0
VEL_MAX_FWD = 900.0
VEL_MAX_BWD = -300.0

STEER_RATE = 2.8

TRACK_WIDTH = 60.0

#Lap Detection
lap = 0
best_lap_time = None
time_in_run = 0.0
s_prev = 0.0


def make_oval_centerline(cx, cy, rx, ry, n=200):
    pts = []
    for i in range(n):
        a = 2*math.pi * i / n
        x = cx + rx * math.cos(a)
        y = cy + ry * math.sin(a)
        pts.append((x, y))
    pts.append(pts[0])
    return pts

def draw_car(surface, x, y, heading, color=(0, 200, 255)):
    L = 34
    Wc = 18
    pts_local = [
        (L/2, 0),
        (-L/2, -Wc/2),
        (-L/2, Wc/2),
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
    pygame.draw.line(surface, (255, 255, 255), (x, y), (nose_x, nose_y))

def dot(ax, ay, bx, by): return ax*bx + ay*by
def length(ax, ay): return math.hypot(ax, ay)

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
    best = None
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
            best = (qx, qy)
            best_i = i
            best_t = t
    return best[0], best[1], best_i, best_t, best_d2

def polyline_prefix_lengths(pts):
    L = [0.0]
    acc = 0.0
    for i in range(len(pts)-1):
        ax, ay = pts[i]
        bx, by = pts[i+1]
        acc += math.hypot(bx-ax, by-ay)
        L.append(acc)
    return L

centerline = make_oval_centerline(W*0.5, H*0.5, 350, 220, n=400)
centerline_lengths = polyline_prefix_lengths(centerline)
track_length = centerline_lengths[-1]


while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

#Logic
    keys = pygame.key.get_pressed()

    throttle  = 0.0
    if keys[pygame.K_UP]:
        throttle += 1.0
    if keys[pygame.K_DOWN]:
        throttle -= 1.0
    throttle = max(-1.0, min(1.0, throttle))

    steer_input = 0.0
    if keys[pygame.K_LEFT]:
        steer_input -= 1.0
    if keys[pygame.K_RIGHT]:
        steer_input += 1.0
    steer_input = max(-1.0, min(1.0, steer_input))

    if throttle > 0:
        acc = throttle * ACCEL
    else:
        acc = throttle * BRAKE

    if abs(vel) > 0.001:
        drag = DRAG * vel
        roll = ROLL * (1 if vel > 0 else -1)
    else:
        drag, roll = 0.0, 0.0

    a_total = acc - drag - roll

    vel += a_total * dt

    if vel > VEL_MAX_FWD:
        vel = VEL_MAX_FWD
    if vel < VEL_MAX_BWD:
        vel = VEL_MAX_BWD

    speed_factor = max(0.1, min(1.0, abs(vel) / 400.0))
    heading += steer_input * STEER_RATE * speed_factor * dt

    x += math.cos(heading) * vel * dt
    y += math.sin(heading) * vel * dt

    margin = 20
    x = max(margin, min(W - margin, x))
    y = max(margin, min(H - margin, y))

    #Metric with track
    qx, qy, seg_i, seg_t, d2 = nearest_point_on_polyline(x, y, centerline)
    ax, ay = centerline[seg_i]
    bx, by = centerline[seg_i+1]
    tx, ty = (bx - ax), (by - ay)
    seg_len = math.hypot(tx, ty) + 1e-9
    nx, ny = tx/seg_len, ty/seg_len
    lx, ly = -ny, nx

    rx, ry = (x - qx), (y - qy)
    lat_error = dot(rx, ry, lx, ly)

    s_at_seg_start = centerline_lengths[seg_i]
    s_here = s_at_seg_start + seg_t * seg_len
    s_here = s_here % track_length

    time_in_run += dt

    crossed_start = (s_here < 5.0 and s_prev > 50.0)
    if crossed_start and not off_track and time_in_run > 1.0:
        lap += 1
        if best_lap_time is None or time_in_run < best_lap_time:
            best_lap_time = time_in_run
        time_in_run = 0.0

    s_prev = s_here

    if keys[pygame.K_r]:
        x, y, = W * 0.5, H * 0.6
        heading = -math.pi / 2
        vel = 0.0
        time_in_run = 0.0
        s_prev = 0.0

    #Off track
    off_track = abs(lat_error) > (TRACK_WIDTH * 0.5)

#View 
    screen.fill(BG)
    
    pygame.draw.lines(screen, (70,90,110), False, centerline, 2)

    step = 8
    left_pts, right_pts = [], []
    for i in range(0, len(centerline)-1, step):
        ax, ay = centerline[i]
        bx, by = centerline[i+1]
        tx, ty = (bx-ax), (by-ay)
        Lseg = (tx*tx + ty*ty) ** 0.5 + 1e-9
        nx, ny = tx / Lseg, ty / Lseg
        lx, ly = -ny, nx
        off = TRACK_WIDTH * 0.5
        left_pts.append((ax + lx*off, ay + ly*off))
        right_pts.append((ax - lx*off, ay - ly*off))

    if len(left_pts) > 1:
        pygame.draw.lines(screen, (40, 130, 40), False, left_pts, 2)
        pygame.draw.lines(screen, (130, 40, 40), False, right_pts, 2)

    draw_car(screen, x, y, heading)

    hud_lines = [
        f"Vel: {vel:7.1f} px/s",
        f"Heading: {math.degrees(heading)%360:6.1f} deg",
        f"lat_error: {lat_error:6.1f} px   off_track: {off_track}",
        f"s: {s_here:7.1f} / {track_length:7.1f}   lap: {lap}",
        f"lap_time: {time_in_run:5.2f}s    best: {(best_lap_time or 0):5.2f}s",
    ]
    for i, line in enumerate(hud_lines):
        txt = font.render(line, True, FG)
        screen.blit(txt, (20, 20 + i *22))

    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False

    pygame.display.flip()

pygame.quit()