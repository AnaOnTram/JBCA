#!/usr/bin/env python3
"""
| File: ld06_udp_listener.py
| Description: Minimal consumer of the simulated LD06 UDP JSON stream (plain python3, no Isaac Sim needed).
|
|     python3 ld06_udp_listener.py              # text summary per scan
|     python3 ld06_udp_listener.py --plot       # live polar plot (needs matplotlib)
|
| Packet fields: stamp, angle_min, angle_increment (rad, CCW from the drone's nose), range_min, range_max,
| ranges[] (m, null = no return), intensities[], sectors_72[] (min distance per 5 deg sector), min_distance.
"""
import argparse
import json
import math
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=9006)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.host, args.port))
print(f"Listening for LD06 scans on udp://{args.host}:{args.port}")


def sector(scan, deg):
    v = scan["sectors_72"][int(round(deg / 5.0)) % 72]
    return "  --  " if v is None else f"{v:6.2f}"


if args.plot:
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure("LD06")
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")  # nose up
    pts = ax.scatter([], [], s=4, c="r")

while True:
    data, _ = sock.recvfrom(65535)
    scan = json.loads(data)
    rng = scan["ranges"]
    valid = sum(r is not None for r in rng)
    md = scan["min_distance"]
    closest = "none" if md is None else f"{md:5.2f} m @ {math.degrees(scan['min_distance_angle']):5.1f} deg"
    print(f"t={scan['stamp']:7.2f}s beams={len(rng)} hits={valid:3d} closest={closest} | "
          f"F{sector(scan, 0)} L{sector(scan, 90)} B{sector(scan, 180)} R{sector(scan, 270)}")

    if args.plot:
        th = [scan["angle_min"] + i * scan["angle_increment"] for i, r in enumerate(rng) if r is not None]
        rr = [r for r in rng if r is not None]
        pts.set_offsets(list(zip(th, rr)) or [(0, 0)])
        ax.set_rmax(scan["range_max"])
        fig.canvas.flush_events()
        plt.pause(0.001)
