"""
| File: urban_city.py
| Description: Procedural urban scene for the Pegasus Simulator (Isaac Sim 5.x).
|
| Builds, directly in USD:
|   - a street grid (asphalt ground, raised sidewalks per block, lane markings)
|   - skyscrapers / mid-rise buildings of random footprint and height on every block
|   - street-light poles and traffic-light poles at intersections (thin obstacles, good lidar targets)
|   - moving vehicles (cars, vans, buses) that drive right-hand traffic loops around city blocks
|
| Every solid object has a PhysX collider, so both the drone and ray-cast sensors (LD06) interact with it.
| Vehicles are kinematic rigid bodies: they push the drone if it gets in the way but are not affected by it.
|
| The world origin (0, 0) is the centre of an intersection so the drone can take off in the open.
"""
import math
import random

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


# --------------------------------------------------------------------------------------------- USD helpers
def _material(stage, path, color, roughness=0.6, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _box(stage, path, center, size, material=None, collider=True, yaw_deg=0.0):
    """Axis-aligned box (optionally yawed) of full extents `size` centred at `center`."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _cylinder(stage, path, center, radius, height, material=None, collider=True, axis="Z"):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(material)
    if collider:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


# --------------------------------------------------------------------------------------------- traffic
class _Vehicle:
    """A kinematic vehicle following a closed polyline at constant speed."""

    def __init__(self, xform, translate_op, rotate_op, waypoints, speed, s0):
        self.xform = xform
        self.translate_op = translate_op
        self.rotate_op = rotate_op
        self.wp = np.asarray(waypoints, dtype=float)
        seg = np.roll(self.wp, -1, axis=0) - self.wp
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_dir = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = self.cum[-1]
        self.speed = speed
        self.s = s0 % self.length
        self.s_initial = self.s

    def pose(self):
        i = int(np.searchsorted(self.cum, self.s, side="right") - 1) % len(self.seg_len)
        along = self.s - self.cum[i]
        p = self.wp[i] + self.seg_dir[i] * along
        yaw = math.atan2(self.seg_dir[i][1], self.seg_dir[i][0])

        # Smooth the heading through the corner over the last/first few metres of each segment
        blend = 4.0
        if along < blend:
            prev = self.seg_dir[i - 1]
            yp = math.atan2(prev[1], prev[0])
            d = (yaw - yp + math.pi) % (2 * math.pi) - math.pi
            yaw = yp + d * (0.5 + 0.5 * along / blend)
        elif self.seg_len[i] - along < blend:
            nxt = self.seg_dir[(i + 1) % len(self.seg_dir)]
            yn = math.atan2(nxt[1], nxt[0])
            d = (yn - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw = yaw + d * 0.5 * (1.0 - (self.seg_len[i] - along) / blend)
        return p, math.degrees(yaw)

    def step(self, dt):
        self.s = (self.s + self.speed * dt) % self.length
        p, yaw = self.pose()
        self.translate_op.Set(Gf.Vec3d(p[0], p[1], 0.0))
        self.rotate_op.Set(float(yaw))


# --------------------------------------------------------------------------------------------- city
class UrbanCity:
    """
    Args:
        stage: the USD stage
        root: prim path under which everything is created
        blocks: number of city blocks along x and y (the grid is centred on the origin)
        block_size: side of a block (m) including its sidewalk
        street_width: width of the road between blocks (m), 2 lanes each way
        num_vehicles: moving vehicles to spawn
        seed: RNG seed, same seed -> same city
    """

    VEHICLE_TYPES = {
        #        length width height  speed range (m/s)       colours
        "car":  (4.5, 1.9, 1.5, (8.0, 13.0), [(0.8, 0.1, 0.1), (0.1, 0.2, 0.7), (0.9, 0.9, 0.9), (0.1, 0.1, 0.1), (0.7, 0.7, 0.75)]),
        "van":  (5.5, 2.1, 2.3, (7.0, 11.0), [(0.95, 0.95, 0.95), (0.2, 0.4, 0.2)]),
        "bus":  (12.0, 2.55, 3.2, (6.0, 9.0), [(0.95, 0.75, 0.1), (0.1, 0.45, 0.8)]),
    }

    def __init__(self, stage, root="/World/City", blocks=6, block_size=40.0, street_width=14.0,
                 num_vehicles=24, seed=7):
        self.stage = stage
        self.root = root
        self.blocks = blocks
        self.block_size = block_size
        self.street_width = street_width
        self.pitch = block_size + street_width
        self.num_vehicles = num_vehicles
        self.rng = random.Random(seed)
        self.vehicles = []
        self.building_boxes = []  # (xmin, xmax, ymin, ymax, height) for reference / spawning checks

    # -- geometry of the grid: street centre lines are at k*pitch, blocks are centred at (k+0.5)*pitch
    def _street_coords(self):
        half = self.blocks // 2
        return [k * self.pitch for k in range(-half, self.blocks - half + 1)]

    def _block_centers(self):
        s = self._street_coords()
        return [0.5 * (s[i] + s[i + 1]) for i in range(len(s) - 1)]

    def build(self):
        stage = self.stage
        UsdGeom.Xform.Define(stage, self.root)
        mats = self.root + "/Looks"
        UsdGeom.Scope.Define(stage, mats)
        self.m_asphalt = _material(stage, mats + "/Asphalt", (0.12, 0.12, 0.13), 0.9)
        self.m_sidewalk = _material(stage, mats + "/Sidewalk", (0.55, 0.55, 0.52), 0.85)
        self.m_marking = _material(stage, mats + "/Marking", (0.95, 0.95, 0.9), 0.5)
        self.m_pole = _material(stage, mats + "/Pole", (0.25, 0.27, 0.3), 0.4, 0.8)
        self.m_glass = [
            _material(stage, mats + f"/Facade{i}", c, r, m) for i, (c, r, m) in enumerate([
                ((0.35, 0.45, 0.55), 0.15, 0.7),   # blue glass
                ((0.55, 0.57, 0.6), 0.3, 0.6),     # steel
                ((0.62, 0.55, 0.45), 0.8, 0.0),    # sandstone
                ((0.3, 0.32, 0.35), 0.2, 0.8),     # dark glass
                ((0.72, 0.7, 0.66), 0.7, 0.0),     # concrete
            ])
        ]
        self.m_vehicle = {}

        self._build_lights()
        self._build_ground()
        self._build_blocks()
        self._build_street_furniture()
        self._build_traffic()
        return self

    def _build_lights(self):
        sun = UsdLux.DistantLight.Define(self.stage, self.root + "/Sun")
        sun.CreateIntensityAttr(3000.0)
        sun.CreateAngleAttr(0.5)
        xf = UsdGeom.Xformable(sun)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 20.0, 35.0))
        dome = UsdLux.DomeLight.Define(self.stage, self.root + "/Sky")
        dome.CreateIntensityAttr(900.0)
        dome.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))

    def _build_ground(self):
        s = self._street_coords()
        extent = (s[-1] - s[0]) + 2 * self.pitch
        # Road surface: its top is z = 0
        _box(self.stage, self.root + "/Ground", (0.0, 0.0, -0.25), (extent, extent, 0.5), self.m_asphalt)

        # Dashed centre lines along every street (visual only)
        g = self.root + "/Markings"
        UsdGeom.Xform.Define(self.stage, g)
        n = 0
        lo, hi = s[0] - self.pitch / 2, s[-1] + self.pitch / 2
        for c in s:
            x = lo
            while x < hi:
                if all(abs(x - k) > self.street_width / 2 + 1.5 for k in s):  # no dashes inside intersections
                    _box(self.stage, f"{g}/d{n}", (x, c, 0.005), (3.0, 0.15, 0.01), self.m_marking, collider=False)
                    _box(self.stage, f"{g}/d{n}b", (c, x, 0.005), (0.15, 3.0, 0.01), self.m_marking, collider=False)
                    n += 1
                x += 6.0

    def _build_blocks(self):
        g = self.root + "/Blocks"
        UsdGeom.Xform.Define(self.stage, g)
        bs = self.block_size
        inset = 3.0  # sidewalk width
        bi = 0
        for cx in self._block_centers():
            for cy in self._block_centers():
                bp = f"{g}/Block_{bi}"
                UsdGeom.Xform.Define(self.stage, bp)
                _box(self.stage, bp + "/Sidewalk", (cx, cy, 0.075), (bs, bs, 0.15), self.m_sidewalk)

                # Downtown (near the centre) is taller
                dist = math.hypot(cx, cy) / (self.pitch * self.blocks / 2)
                h_max = 180.0 * (1.0 - 0.65 * min(dist, 1.0))
                lot = bs - 2 * inset

                # Either one tower filling the block, or a 2x2 subdivision of lots
                if self.rng.random() < 0.3:
                    lots = [(cx, cy, lot, lot)]
                else:
                    q = lot / 2
                    lots = [(cx + dx * q / 2, cy + dy * q / 2, q, q) for dx in (-1, 1) for dy in (-1, 1)]

                for li, (lx, ly, w, d) in enumerate(lots):
                    if self.rng.random() < 0.12:
                        continue  # empty lot / plaza
                    fw = w * self.rng.uniform(0.85, 1.0)
                    fd = d * self.rng.uniform(0.85, 1.0)
                    h = self.rng.uniform(15.0, max(25.0, h_max))
                    mat = self.rng.choice(self.m_glass)
                    base = f"{bp}/Building_{li}"
                    UsdGeom.Xform.Define(self.stage, base)
                    _box(self.stage, base + "/Tower", (lx, ly, 0.15 + h / 2), (fw, fd, h), mat)
                    self.building_boxes.append((lx - fw / 2, lx + fw / 2, ly - fd / 2, ly + fd / 2, h))
                    # Setback crown on tall towers
                    if h > 60.0 and self.rng.random() < 0.6:
                        ch = self.rng.uniform(8.0, 25.0)
                        _box(self.stage, base + "/Crown", (lx, ly, 0.15 + h + ch / 2), (fw * 0.6, fd * 0.6, ch), mat)
                        if self.rng.random() < 0.5:
                            _cylinder(self.stage, base + "/Antenna", (lx, ly, 0.15 + h + ch + 6.0), 0.3, 12.0, self.m_pole)
                bi += 1

    def _build_street_furniture(self):
        """Street lights along sidewalks and traffic-light poles at intersection corners."""
        g = self.root + "/Furniture"
        UsdGeom.Xform.Define(self.stage, g)
        s = self._street_coords()
        off = self.street_width / 2 + 0.6  # just on the sidewalk edge
        n = 0
        for x in s:
            for y in s:
                for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    px, py = x + sx * off, y + sy * off
                    _cylinder(self.stage, f"{g}/TL_{n}", (px, py, 2.5), 0.12, 5.0, self.m_pole)
                    # Mast arm over the road
                    _box(self.stage, f"{g}/TLArm_{n}", (px - sx * 2.0, py, 5.0), (4.0, 0.15, 0.15), self.m_pole)
                    n += 1
        # Street lights mid-block on both sides of every street
        for c in s:
            for b in self._block_centers():
                for side in (1, -1):
                    for along in (-self.block_size / 4, self.block_size / 4):
                        _cylinder(self.stage, f"{g}/SL_{n}", (b + along, c + side * off, 4.0), 0.1, 8.0, self.m_pole)
                        _cylinder(self.stage, f"{g}/SL_{n}b", (c + side * off, b + along, 4.0), 0.1, 8.0, self.m_pole)
                        n += 1

    # ------------------------------------------------------------------ traffic
    def _vehicle_material(self, color):
        key = tuple(round(c, 2) for c in color)
        if key not in self.m_vehicle:
            self.m_vehicle[key] = _material(self.stage, self.root + f"/Looks/Paint{len(self.m_vehicle)}", color, 0.25, 0.6)
        return self.m_vehicle[key]

    def _make_vehicle_prim(self, path, kind):
        L, W, H, _, colours = self.VEHICLE_TYPES[kind]
        paint = self._vehicle_material(self.rng.choice(colours))
        xform = UsdGeom.Xform.Define(self.stage, path)
        t_op = xform.AddTranslateOp()
        r_op = xform.AddRotateZOp()
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
        clearance = 0.3
        if kind == "bus":
            _box(self.stage, path + "/Body", (0, 0, clearance + (H - clearance) / 2), (L, W, H - clearance), paint)
        else:
            body_h = (H - clearance) * 0.55
            _box(self.stage, path + "/Body", (0, 0, clearance + body_h / 2), (L, W, body_h), paint)
            cab_l = L * (0.55 if kind == "car" else 0.75)
            cab_x = -L * 0.05 if kind == "car" else -L * 0.1
            _box(self.stage, path + "/Cabin", (cab_x, 0, clearance + body_h + (H - clearance - body_h) / 2),
                 (cab_l, W * 0.92, H - clearance - body_h), self.m_glass[3])
        for i, (wx, wy) in enumerate(((L * 0.35, W / 2), (L * 0.35, -W / 2), (-L * 0.35, W / 2), (-L * 0.35, -W / 2))):
            _cylinder(self.stage, path + f"/Wheel{i}", (wx, wy, 0.35), 0.35, 0.25, self.m_asphalt, collider=False, axis="Y")
        return xform, t_op, r_op

    def _loop_waypoints(self, x0, x1, y0, y1, clockwise):
        """Rectangle along street centre lines, offset into the right-hand lane for the travel direction."""
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # counter-clockwise
        if clockwise:
            corners = corners[::-1]
        pts = np.array(corners, dtype=float)
        n = len(pts)
        out = []
        lane = self.street_width / 4  # centre of the outer lane... inner lane for opposite direction
        lane_off = self.rng.choice([lane * 0.6, lane * 1.4])  # pick one of the two lanes on our side
        for i in range(n):
            d_in = pts[i] - pts[i - 1]
            d_out = pts[(i + 1) % n] - pts[i]
            d_in /= np.linalg.norm(d_in)
            d_out /= np.linalg.norm(d_out)
            r_in = np.array([d_in[1], -d_in[0]])
            r_out = np.array([d_out[1], -d_out[0]])
            out.append(pts[i] + lane_off * (r_in + r_out))  # miter offset for right angles
        return out

    def _build_traffic(self):
        g = self.root + "/Traffic"
        UsdGeom.Xform.Define(self.stage, g)
        s = self._street_coords()
        kinds = ["car"] * 6 + ["van"] * 2 + ["bus"]
        for v in range(self.num_vehicles):
            # Random rectangular loop spanning 1..3 blocks in each direction
            i0 = self.rng.randrange(0, len(s) - 1)
            j0 = self.rng.randrange(0, len(s) - 1)
            i1 = min(len(s) - 1, i0 + self.rng.randint(1, 3))
            j1 = min(len(s) - 1, j0 + self.rng.randint(1, 3))
            wps = self._loop_waypoints(s[i0], s[i1], s[j0], s[j1], clockwise=self.rng.random() < 0.5)
            kind = self.rng.choice(kinds)
            speed = self.rng.uniform(*self.VEHICLE_TYPES[kind][3])
            xform, t_op, r_op = self._make_vehicle_prim(f"{g}/{kind.capitalize()}_{v}", kind)
            veh = _Vehicle(xform, t_op, r_op, wps, speed, s0=self.rng.uniform(0, 1e4))
            veh.step(0.0)
            self.vehicles.append(veh)

    # ------------------------------------------------------------------ runtime
    def step(self, dt):
        """Advance every vehicle. Call from a physics callback."""
        with Sdf.ChangeBlock():
            for v in self.vehicles:
                v.step(dt)

    def reset(self):
        for v in self.vehicles:
            v.s = v.s_initial
            v.step(0.0)
