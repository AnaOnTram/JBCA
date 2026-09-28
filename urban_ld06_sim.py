#!/usr/bin/env python
"""
| File: urban_ld06_sim.py
| Description: Pegasus Simulator app - Iris quadrotor with an LD06 360 deg 2D lidar flying in a procedural
|              downtown (skyscrapers, poles, moving traffic). PX4 SITL is auto-launched and controls the drone.
|
| Run with the Isaac Sim python interpreter:
|     ~/isaacsim/python.sh urban_ld06_sim.py [--headless] [--no-px4] [--mavlink-obstacles] [--ros2] [--visualize]
|
| LD06 distance data leaves the simulator through:
|   * UDP JSON on 127.0.0.1:9006 (always)                -> python3 ld06_udp_listener.py
|   * MAVLink OBSTACLE_DISTANCE to PX4 (--mavlink-obstacles) -> PX4 collision prevention (set CP_DIST)
|   * ROS 2 sensor_msgs/LaserScan on /iris0/scan (--ros2)
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Pegasus urban scene with an LD06-equipped Iris")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-px4", action="store_true", help="do not attach/launch PX4 (drone stays idle on the ground)")
parser.add_argument("--mavlink-obstacles", action="store_true", help="send OBSTACLE_DISTANCE to PX4 SITL")
parser.add_argument("--ros2", action="store_true", help="publish sensor_msgs/LaserScan")
parser.add_argument("--visualize", action="store_true", help="draw LD06 hit points in the viewport")
parser.add_argument("--udp-port", type=int, default=9006)
parser.add_argument("--vehicles", type=int, default=24, help="number of moving vehicles")
parser.add_argument("--blocks", type=int, default=6, help="city blocks per side")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--max-steps", type=int, default=0, help="exit after N physics steps (0 = run until closed)")
args, _ = parser.parse_known_args()

# Isaac Sim must be started before any omni/pxr/pegasus import
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
import omni.usd
from isaacsim.core.api.world import World
from isaacsim.core.utils.viewports import set_camera_view
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Local modules (this folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urban_city import UrbanCity
from ld06 import LD06Lidar, LD06OutputBackend


class UrbanLD06App:

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        stage = omni.usd.get_context().get_stage()

        # ---------------- Urban environment ----------------
        self.city = UrbanCity(stage, "/World/City", blocks=args.blocks, num_vehicles=args.vehicles, seed=args.seed).build()
        self.world.add_physics_callback("city_traffic", self.city.step)
        carb.log_warn(f"[Urban] built {len(self.city.building_boxes)} buildings and {len(self.city.vehicles)} moving vehicles")

        # ---------------- Iris + LD06 ----------------
        config = MultirotorConfig()

        # Keep the default IMU / GPS / baro / mag and add the LD06 on top of the airframe
        self.ld06 = LD06Lidar({
            "frame_id": "ld06_link",
            "position": [0.0, 0.0, 0.10],   # FLU body frame, above the propeller plane
            "scan_frequency": 10.0,
            "sample_rate": 4500.0,
            "min_range": 0.02,
            "max_range": 12.0,
            "visualize": args.visualize,
        })
        config.sensors = config.sensors + [self.ld06]

        self.ld06_out = LD06OutputBackend(vehicle_id=0, config={
            "udp": True,
            "udp_port": args.udp_port,
            "mavlink": args.mavlink_obstacles and not args.no_px4,
            "ros2": args.ros2,
            "print_rate": 1.0,
        })

        backends = []
        if not args.no_px4:
            mavlink_config = PX4MavlinkBackendConfig({
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
            })
            backends.append(PX4MavlinkBackend(mavlink_config))  # MUST be first: it drives the rotors
        backends.append(self.ld06_out)
        config.backends = backends

        # Spawn in the middle of the central intersection (between the traffic lanes)
        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        set_camera_view(eye=[-18.0, -18.0, 12.0], target=[0.0, 0.0, 2.0])
        self.world.reset()

    def run(self):
        self.timeline.play()
        steps = 0
        while simulation_app.is_running() and (args.max_steps == 0 or steps < args.max_steps):
            self.world.step(render=True)
            steps += 1

        carb.log_warn("Urban LD06 app is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    UrbanLD06App().run()


if __name__ == "__main__":
    main()
