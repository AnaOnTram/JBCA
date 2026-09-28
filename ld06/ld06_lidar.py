"""
| File: ld06_lidar.py
| Description: Simulated LDROBOT LD06 360-degree 2D laser range finder for the Pegasus Simulator.
|
| The LD06 is a spinning DTOF range finder:
|   - 360 deg horizontal field of view, single scan plane
|   - 4500 samples/s, 10 Hz nominal rotation  -> ~450 beams per revolution (0.8 deg step)
|   - 0.02 m .. 12 m range
|   - accuracy roughly +/-10 mm (< 300 mm) and +/-45 mm further out
|
| The sensor is implemented with PhysX scene-query raycasts, so it "sees" every prim that has a collider
| (buildings, moving vehicles, poles, ...). It emulates the real rotating head: on every physics step only the
| beams that were swept since the previous step are cast, which also reproduces the motion skew a real spinning
| lidar has when the drone moves or rotates during a revolution. A full scan is emitted once per revolution.
|
| Output (the dict returned by update() and forwarded to every backend with sensor_type == "LD06"):
|   ranges          np.ndarray (N,)   distance in metres, np.inf when nothing was hit inside range
|   intensities     np.ndarray (N,)   0..255, simple incidence-angle based return strength (0 when no hit)
|   angles          np.ndarray (N,)   beam angle in radians, body FLU frame, 0 = forward, counter-clockwise positive
|   angle_min / angle_max / angle_increment, range_min / range_max, scan_time, time_increment
|   sectors_72      np.ndarray (72,)  minimum distance in each 5 deg sector (inf if empty), sector 0 centred on forward
|   timestamp       float             simulation time (s) of the end of the revolution
|   frame_id        str
"""
__all__ = ["LD06Lidar"]

import numpy as np
from scipy.spatial.transform import Rotation

import carb
from omni.physx import get_physx_scene_query_interface

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor


class LD06Lidar(Sensor):

    def __init__(self, config={}):
        """
        Args:
            config (dict): Optional parameters. Defaults:

            >>> {"frame_id": "ld06_link",
            >>>  "position": [0.0, 0.0, 0.09],     # mount position in the vehicle body frame (FLU, m)
            >>>  "yaw_offset_deg": 0.0,            # mount yaw of the sensor w.r.t. the body x axis
            >>>  "scan_frequency": 10.0,           # revolutions per second (LD06: 5-13 Hz, nominal 10)
            >>>  "sample_rate": 4500.0,            # range samples per second
            >>>  "min_range": 0.02,
            >>>  "max_range": 12.0,
            >>>  "noise_std_near": 0.010,          # 1-sigma range noise below 0.3 m
            >>>  "noise_std_far": 0.015,           # 1-sigma range noise above 0.3 m (~ +/-45 mm at 3 sigma)
            >>>  "dropout_prob": 0.0,              # probability that a valid return is lost
            >>>  "ignore_self": True,              # ignore hits on the vehicle carrying the sensor
            >>>  "visualize": False}               # draw the hit points of every scan in the viewport
        """
        self._scan_frequency = float(config.get("scan_frequency", 10.0))
        self._sample_rate = float(config.get("sample_rate", 4500.0))

        # The base class rate limiter is not used (the sweep is advanced on every physics step), but keep the
        # nominal scan rate so that sensor.update_rate reports something meaningful
        super().__init__(sensor_type="LD06", update_rate=self._scan_frequency)

        self._frame_id = config.get("frame_id", "ld06_link")
        self._position = np.array(config.get("position", [0.0, 0.0, 0.09]), dtype=float)
        self._yaw_offset = np.radians(config.get("yaw_offset_deg", 0.0))
        self._min_range = float(config.get("min_range", 0.02))
        self._max_range = float(config.get("max_range", 12.0))
        self._noise_near = float(config.get("noise_std_near", 0.010))
        self._noise_far = float(config.get("noise_std_far", 0.015))
        self._dropout_prob = float(config.get("dropout_prob", 0.0))
        self._ignore_self = bool(config.get("ignore_self", True))
        self._visualize = bool(config.get("visualize", False))

        # Beam layout of one revolution
        self._num_beams = int(round(self._sample_rate / self._scan_frequency))
        self._angle_increment = 2.0 * np.pi / self._num_beams
        self._angles = np.arange(self._num_beams) * self._angle_increment
        # Unit beam directions in the sensor frame (planar)
        self._beam_dirs = np.stack([np.cos(self._angles + self._yaw_offset),
                                    np.sin(self._angles + self._yaw_offset),
                                    np.zeros(self._num_beams)], axis=1)

        self._rng = np.random.default_rng()
        self._draw = None
        self._reset_sweep()
        self._state = None

    # ------------------------------------------------------------------ helpers
    def _reset_sweep(self):
        self._ranges = np.full(self._num_beams, np.inf)
        self._intensities = np.zeros(self._num_beams)
        self._next_beam = 0
        self._sample_accum = 0.0
        self._sim_time = 0.0

    def _cast(self, sq, origin, direction):
        """Return (distance, cos_incidence) of the closest non-self hit, or (inf, 0)."""
        o = carb.Float3(*origin)
        d = carb.Float3(*direction)

        # Fast path: the closest hit is usually not the airframe (the sensor sits above it)
        hit = sq.raycast_closest(o, d, self._max_range)
        if not hit["hit"]:
            return np.inf, 0.0
        own = hit["rigidBody"].startswith(self._vehicle_path) or hit["collision"].startswith(self._vehicle_path)
        if not (self._ignore_self and own):
            n = hit["normal"]
            return hit["distance"], abs(n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2])

        # The ray hit our own vehicle first: collect every hit and keep the closest foreign one
        best = [np.inf, 0.0]
        prefix = self._vehicle_path

        def report(h):
            if h.rigid_body.startswith(prefix) or h.collision.startswith(prefix):
                return True  # own airframe, keep looking
            if h.distance < best[0]:
                n = h.normal
                best[0] = h.distance
                best[1] = abs(n.x * direction[0] + n.y * direction[1] + n.z * direction[2])
            return True

        sq.raycast_all(o, d, self._max_range, report)
        return best[0], best[1]

    def _add_noise(self, r):
        if not np.isfinite(r):
            return r
        sigma = self._noise_near if r < 0.3 else self._noise_far
        return r + self._rng.normal(0.0, sigma)

    # ------------------------------------------------------------------ Sensor API
    def initialize(self, vehicle, origin_lat, origin_lon, origin_alt):
        super().initialize(vehicle, origin_lat, origin_lon, origin_alt)
        self._vehicle_path = vehicle.prim_path

    def start(self):
        self._reset_sweep()
        if self._visualize:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self._draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as e:  # pragma: no cover - visualisation is optional
                carb.log_warn(f"[LD06] debug draw unavailable, disabling visualisation: {e}")
                self._draw = None

    def stop(self):
        if self._draw is not None:
            self._draw.clear_points()

    def reset(self):
        self._reset_sweep()

    @property
    def state(self):
        return self._state

    def update(self, state: State, dt: float):
        """Advance the rotating head by dt. Returns a full scan dict once per revolution, otherwise None."""
        self._sim_time += dt
        self._sample_accum += self._sample_rate * dt
        n_new = int(self._sample_accum)
        if n_new == 0:
            return None
        self._sample_accum -= n_new

        sq = get_physx_scene_query_interface()

        # Pose of the sensor in the world (ENU). state.attitude is [qx, qy, qz, qw] of body FLU w.r.t. ENU
        rot = Rotation.from_quat(state.attitude)
        origin = state.position + rot.apply(self._position)
        R = rot.as_matrix()

        result = None
        for _ in range(n_new):
            i = self._next_beam
            direction = R @ self._beam_dirs[i]
            dist, cos_inc = self._cast(sq, origin, direction)

            if dist < self._min_range or (self._dropout_prob > 0.0 and self._rng.random() < self._dropout_prob):
                dist = np.inf
            dist = self._add_noise(dist)
            if np.isfinite(dist):
                dist = float(np.clip(dist, self._min_range, self._max_range))
                # Crude return strength: stronger when close and when hitting the surface head on
                self._intensities[i] = np.clip(255.0 * cos_inc * (1.0 - 0.5 * dist / self._max_range), 10.0, 255.0)
            else:
                self._intensities[i] = 0.0
            self._ranges[i] = dist

            self._next_beam += 1
            if self._next_beam >= self._num_beams:
                result = self._publish_scan(origin, R)
                self._next_beam = 0
        return result

    def _publish_scan(self, origin, R):
        ranges = self._ranges.copy()

        # Minimum distance per 5 deg sector, sector k centred on k*5 deg (CCW from forward)
        sector_idx = (np.round(np.degrees(self._angles) / 5.0).astype(int)) % 72
        sectors = np.full(72, np.inf)
        np.minimum.at(sectors, sector_idx, ranges)

        self._state = {
            "frame_id": self._frame_id,
            "timestamp": self._sim_time,
            "angle_min": 0.0,
            "angle_max": float(self._angles[-1]),
            "angle_increment": float(self._angle_increment),
            "time_increment": 1.0 / self._sample_rate,
            "scan_time": 1.0 / self._scan_frequency,
            "range_min": self._min_range,
            "range_max": self._max_range,
            "angles": self._angles,
            "ranges": ranges,
            "intensities": self._intensities.copy(),
            "sectors_72": sectors,
            "min_distance": float(np.min(ranges)),
            "min_distance_angle": float(self._angles[int(np.argmin(ranges))]),
        }

        if self._draw is not None:
            valid = np.isfinite(ranges)
            if np.any(valid):
                pts = origin + (self._beam_dirs[valid] * ranges[valid, None]) @ R.T
                self._draw.clear_points()
                self._draw.draw_points([tuple(p) for p in pts], [(1.0, 0.1, 0.1, 1.0)] * len(pts), [6.0] * len(pts))

        return self._state
