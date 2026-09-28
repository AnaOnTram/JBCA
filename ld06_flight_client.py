#!/usr/bin/env python3
"""
| File: ld06_flight_client.py
| Description: Read the simulated LD06 laser scan (UDP JSON from the simulator) and the drone's flight state
|              (MAVLink telemetry from PX4 SITL) together in one Python process.
|
| Needs only pymavlink (no Isaac Sim, no ROS). Either:
|     ~/isaacsim/python.sh ld06_flight_client.py            # Isaac's python already has pymavlink
| or
|     pip install pymavlink && python3 ld06_flight_client.py
|
| Use it as a library:
|     from ld06_flight_client import LD06FlightClient
|     c = LD06FlightClient().start()
|     scan, flight = c.latest()
|
| Frames:
|   scan angles  body frame, 0 = nose, counter-clockwise positive (ROS LaserScan convention)
|   flight       PX4 local NED: x = north, y = east, z = down (m); yaw from north, clockwise (rad)
"""
import json
import math
import socket
import threading
import time

from pymavlink import mavutil


class LD06FlightClient:

    def __init__(self, scan_port=9006, mavlink_url="udpin:0.0.0.0:14540"):
        """
        Args:
            scan_port: UDP port the simulator streams LD06 scans to (urban_ld06_sim.py --udp-port)
            mavlink_url: PX4 telemetry. 14540 is the PX4 SITL offboard/API link. Only one process can bind a
                         port: if MAVSDK/MAVROS already uses 14540, use "udpin:0.0.0.0:14550" (the QGC link)
                         with QGroundControl closed.
        """
        self._scan_port = scan_port
        self._mavlink_url = mavlink_url
        self._lock = threading.Lock()
        self._scan = None
        self._flight = {}
        self.mav = None  # pymavlink connection, usable to send commands as well

    # ------------------------------------------------------------------ public API
    def start(self):
        # Bind here (not in the thread) so a busy port fails loudly instead of silently giving no scans
        self._scan_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._scan_sock.bind(("0.0.0.0", self._scan_port))
        except OSError as e:
            raise RuntimeError(
                f"UDP port {self._scan_port} is busy ({e}). Another program is already receiving LD06 scans, "
                f"e.g. ld06_udp_listener.py. Stop it (find it with: ss -ulpn | grep {self._scan_port}), or start "
                f"the simulator with --udp-port <other> and pass scan_port=<other> here.") from e
        threading.Thread(target=self._scan_loop, daemon=True).start()
        self.mav = mavutil.mavlink_connection(self._mavlink_url)
        threading.Thread(target=self._mavlink_loop, daemon=True).start()
        return self

    def latest(self):
        """Return (scan, flight): the newest LaserScan-like dict and the flight-state dict (either may be None/{})."""
        with self._lock:
            return self._scan, dict(self._flight)

    def obstacle_points_ned(self):
        """Project the latest scan hits into the local NED frame using the latest position/yaw (planar, ignores
        roll/pitch). Returns a list of (north, east) points in metres."""
        scan, fl = self.latest()
        if scan is None or "north" not in fl or "yaw" not in fl:
            return []
        pts = []
        for i, r in enumerate(scan["ranges"]):
            if r is None:
                continue
            a = scan["angle_min"] + i * scan["angle_increment"]  # CCW from the nose
            bearing = fl["yaw"] - a                              # clockwise from north
            pts.append((fl["north"] + r * math.cos(bearing), fl["east"] + r * math.sin(bearing)))
        return pts

    # ------------------------------------------------------------------ receivers
    def _scan_loop(self):
        sock = self._scan_sock
        while True:
            data, _ = sock.recvfrom(65535)
            scan = json.loads(data)
            scan["received"] = time.time()
            with self._lock:
                self._scan = scan

    def _mavlink_loop(self):
        wanted = ["HEARTBEAT", "LOCAL_POSITION_NED", "ATTITUDE", "GLOBAL_POSITION_INT", "SYS_STATUS", "VFR_HUD"]
        while True:
            msg = self.mav.recv_match(type=wanted, blocking=True, timeout=1.0)
            if msg is None:
                continue
            t = msg.get_type()
            upd = {}
            if t == "HEARTBEAT" and msg.get_srcComponent() == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
                upd["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                upd["mode"] = self.mav.flightmode
            elif t == "LOCAL_POSITION_NED":
                upd.update(north=msg.x, east=msg.y, down=msg.z, vn=msg.vx, ve=msg.vy, vd=msg.vz)
            elif t == "ATTITUDE":
                upd.update(roll=msg.roll, pitch=msg.pitch, yaw=msg.yaw)
            elif t == "GLOBAL_POSITION_INT":
                upd.update(lat=msg.lat / 1e7, lon=msg.lon / 1e7, alt_amsl=msg.alt / 1000.0,
                           alt_rel=msg.relative_alt / 1000.0)
            elif t == "SYS_STATUS":
                upd["battery_pct"] = msg.battery_remaining
            elif t == "VFR_HUD":
                upd.update(groundspeed=msg.groundspeed, heading_deg=msg.heading)
            if upd:
                upd["flight_received"] = time.time()
                with self._lock:
                    self._flight.update(upd)


def _fmt(v, f="{:6.2f}"):
    return "   -- " if v is None else f.format(v)


if __name__ == "__main__":
    c = LD06FlightClient().start()
    print("Waiting for LD06 scans on UDP 9006 and PX4 telemetry on UDP 14540 ...")
    while True:
        time.sleep(0.5)
        scan, fl = c.latest()
        s72 = scan["sectors_72"] if scan else [None] * 72
        line = (f"[flight] {fl.get('mode', '?'):>10} armed={fl.get('armed', '?')!s:5} "
                f"N={_fmt(fl.get('north'))} E={_fmt(fl.get('east'))} alt={_fmt(-fl['down'] if 'down' in fl else None)} "
                f"yaw={_fmt(math.degrees(fl['yaw']) if 'yaw' in fl else None, '{:6.1f}')}deg")
        if scan:
            line += (f" | [scan t={scan['stamp']:.1f}s] closest={_fmt(scan['min_distance'])} m "
                     f"F={_fmt(s72[0])} L={_fmt(s72[18])} B={_fmt(s72[36])} R={_fmt(s72[54])} "
                     f"world_pts={len(c.obstacle_points_ned())}")
        else:
            line += " | [scan] none yet"
        print(line, flush=True)
