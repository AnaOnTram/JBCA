"""
| File: ld06_backend.py
| Description: Pegasus backend that exposes the LD06 range data to the outside world.
|
| Three independent outputs (enable any combination):
|   udp     - one JSON datagram per scan (no dependencies, works with any language). Default ON.
|   mavlink - MAVLink OBSTACLE_DISTANCE (72 x 5 deg sectors, cm) sent to PX4 SITL, which feeds PX4's
|             collision prevention / obstacle map exactly like the real LD06 driver does on hardware.
|   ros2    - sensor_msgs/LaserScan through the Isaac Sim ROS 2 bridge.
|
| This backend never controls the vehicle. Pegasus takes rotor commands from the vehicle's first backend, so put
| the flight-controller backend (e.g. PX4MavlinkBackend) first; if this one is first the rotors stay stopped.
"""
__all__ = ["LD06OutputBackend"]

import json
import socket
import time

import numpy as np
import carb

from pegasus.simulator.logic.backends.backend import Backend


class LD06OutputBackend(Backend):

    def __init__(self, vehicle_id: int = 0, config: dict = {}):
        """
        Args:
            vehicle_id (int): used to pick the default PX4 port (14280 + id) and topic namespace.
            config (dict): defaults

            >>> {"udp": True, "udp_host": "127.0.0.1", "udp_port": 9006,
            >>>  "mavlink": False, "mavlink_host": "127.0.0.1", "mavlink_port": 14280 + id,
            >>>  "ros2": False, "ros2_topic": "/iris<id>/scan",
            >>>  "print_rate": 1.0}                           # Hz, console summary (0 disables)
        """
        super().__init__(config)
        self._id = vehicle_id

        self._udp_enabled = config.get("udp", True)
        self._udp_addr = (config.get("udp_host", "127.0.0.1"), int(config.get("udp_port", 9006)))
        self._udp_sock = None

        self._mav_enabled = config.get("mavlink", False)
        # PX4 SITL "onboard payload" MAVLink instance. Not 14580: PX4 replies to the last sender on a port, so
        # writing there would steal the telemetry stream from MAVSDK / MAVROS / QGC-onboard users on 14540
        self._mav_addr = (config.get("mavlink_host", "127.0.0.1"), int(config.get("mavlink_port", 14280 + vehicle_id)))
        self._mav = None
        self._mav_sock = None

        self._ros_enabled = config.get("ros2", False)
        self._ros_topic = config.get("ros2_topic", f"/iris{vehicle_id}/scan")
        self._ros_node = None
        self._ros_pub = None

        self._print_period = 1.0 / config["print_rate"] if config.get("print_rate", 1.0) > 0 else None
        self._last_print = 0.0
        self._t0 = time.time()

        # Latest scan, also useful when the backend is polled from Python directly
        self.latest_scan = None

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self._t0 = time.time()
        if self._udp_enabled and self._udp_sock is None:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            carb.log_warn(f"[LD06] streaming scans as JSON over UDP to {self._udp_addr[0]}:{self._udp_addr[1]}")

        if self._mav_enabled and self._mav is None:
            # OBSTACLE_DISTANCE (id 330) only exists in MAVLink 2, while Pegasus loads pymavlink's global
            # mavutil in MAVLink 1 mode, so encode with the v2 dialect directly and send over a plain UDP socket
            from pymavlink.dialects.v20 import common as mavlink2
            self._mavlink2 = mavlink2
            self._mav = mavlink2.MAVLink(None, srcSystem=1, srcComponent=mavlink2.MAV_COMP_ID_OBSTACLE_AVOIDANCE)
            self._mav_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            carb.log_warn(f"[LD06] sending MAVLink OBSTACLE_DISTANCE to udp://{self._mav_addr[0]}:{self._mav_addr[1]}")

        if self._ros_enabled and self._ros_node is None:
            try:
                from isaacsim.core.utils.extensions import enable_extension
                enable_extension("isaacsim.ros2.bridge")
                import rclpy
                from rclpy.qos import qos_profile_sensor_data
                from sensor_msgs.msg import LaserScan
                try:
                    rclpy.init()
                except Exception:
                    pass  # already initialised (e.g. by the Pegasus ROS2Backend)
                self._LaserScan = LaserScan
                self._ros_node = rclpy.create_node(f"ld06_lidar_{self._id}")
                self._ros_pub = self._ros_node.create_publisher(LaserScan, self._ros_topic, qos_profile_sensor_data)
                carb.log_warn(f"[LD06] publishing sensor_msgs/LaserScan on {self._ros_topic}")
            except Exception as e:
                carb.log_error(f"[LD06] ROS 2 output unavailable: {e}")
                self._ros_node = None

    def stop(self):
        if self._udp_sock is not None:
            self._udp_sock.close()
            self._udp_sock = None
        if self._mav_sock is not None:
            self._mav_sock.close()
            self._mav_sock = None
            self._mav = None
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None

    def reset(self):
        pass

    # ------------------------------------------------------------------ data path
    def update_sensor(self, sensor_type: str, data):
        if sensor_type != "LD06" or data is None:
            return
        self.latest_scan = data

        if self._udp_sock is not None:
            self._send_udp(data)
        if self._mav is not None:
            self._send_mavlink(data)
        if self._ros_pub is not None:
            self._send_ros2(data)

        if self._print_period is not None and data["timestamp"] - self._last_print >= self._print_period:
            self._last_print = data["timestamp"]
            s = data["sectors_72"]
            f = lambda deg: s[(deg // 5) % 72]
            carb.log_warn(
                f"[LD06] t={data['timestamp']:.1f}s  closest {data['min_distance']:.2f} m @ "
                f"{np.degrees(data['min_distance_angle']):.0f} deg | front {f(0):.2f}  left {f(90):.2f}  "
                f"back {f(180):.2f}  right {f(270):.2f}")

    def _send_udp(self, d):
        # inf is not valid JSON -> encode "no return" as null
        def clean(a):
            return [None if not np.isfinite(v) else round(float(v), 4) for v in a]

        msg = {
            "sensor": "LD06",
            "vehicle_id": self._id,
            "frame_id": d["frame_id"],
            "stamp": d["timestamp"],
            "angle_min": d["angle_min"],
            "angle_increment": d["angle_increment"],
            "range_min": d["range_min"],
            "range_max": d["range_max"],
            "scan_time": d["scan_time"],
            "ranges": clean(d["ranges"]),
            "intensities": [int(v) for v in d["intensities"]],
            "sectors_72": clean(d["sectors_72"]),
            "min_distance": d["min_distance"] if np.isfinite(d["min_distance"]) else None,
            "min_distance_angle": d["min_distance_angle"],
        }
        try:
            self._udp_sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self._udp_addr)
        except OSError as e:
            carb.log_warn(f"[LD06] UDP send failed: {e}")

    def _send_mavlink(self, d):
        mav2 = self._mavlink2
        # OBSTACLE_DISTANCE uses a clockwise angle convention in MAV_FRAME_BODY_FRD; our sectors are CCW (FLU).
        # Sector k of the message (angle_offset + k*5 deg clockwise) == CCW sector (-k) mod 72.
        ccw = d["sectors_72"]
        cw = ccw[(-np.arange(72)) % 72]
        min_cm = int(d["range_min"] * 100)
        max_cm = int(d["range_max"] * 100)
        # UINT16_MAX = unknown, max_cm + 1 = no obstacle
        dist = [max_cm + 1 if not np.isfinite(v) else int(np.clip(v * 100, min_cm, max_cm)) for v in cw]
        try:
            msg = self._mav.obstacle_distance_encode(
                int(d["timestamp"] * 1e6),
                mav2.MAV_DISTANCE_SENSOR_LASER,
                dist,
                5,            # increment (deg)
                min_cm,
                max_cm,
                0.0,          # increment_f (use integer increment)
                0.0,          # angle_offset: element 0 is centred on forward
                mav2.MAV_FRAME_BODY_FRD,
            )
            self._mav_sock.sendto(msg.pack(self._mav), self._mav_addr)
        except OSError as e:
            carb.log_warn(f"[LD06] MAVLink send failed: {e}")

    def _send_ros2(self, d):
        msg = self._LaserScan()
        t = d["timestamp"]
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int((t - int(t)) * 1e9)
        msg.header.frame_id = d["frame_id"]
        msg.angle_min = float(d["angle_min"])
        msg.angle_max = float(d["angle_max"])
        msg.angle_increment = float(d["angle_increment"])
        msg.time_increment = float(d["time_increment"])
        msg.scan_time = float(d["scan_time"])
        msg.range_min = float(d["range_min"])
        msg.range_max = float(d["range_max"])
        msg.ranges = d["ranges"].astype(np.float32).tolist()
        msg.intensities = d["intensities"].astype(np.float32).tolist()
        self._ros_pub.publish(msg)

    # ------------------------------------------------------------------ unused Backend hooks
    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state):
        pass

    def input_reference(self):
        # Only used if this is the vehicle's first backend (e.g. no PX4): keep the rotors stopped
        return [0.0, 0.0, 0.0, 0.0]

    def update(self, dt: float):
        pass
