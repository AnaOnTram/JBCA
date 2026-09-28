#!/usr/bin/env python3
"""
| File: jev_collision_guard.py
| Description: Collision-stop decisions for the LD06-equipped Iris using TypeSafe's System One model Jev.
|
| Pipeline (one decision):
|   LD06 scan + PX4 flight state
|     -> code: geometry (8 directions, closing speed, time to contact, what lies in the flight path)
|     -> code: hard safety rule (always stops, independent of the model)
|     -> Jev:  typed questions over a worded state   -> stop probability, recommended action, threat direction
|     -> code: combine rule + Jev into a StopDecision (optionally command PX4 to HOLD)
|
| Why the split: Jev's docs say to keep known rules, arithmetic and numeric comparisons in code and to give the
| model semantic descriptions rather than raw numbers ("close", "approaching", "contact imminent").
|
| Run (inside the `jev` conda env, with the simulator running):
|     python jev_collision_guard.py                 # print decisions
|     python jev_collision_guard.py --act           # also switch PX4 to HOLD when a stop is decided
|     python jev_collision_guard.py --dry-run       # no API calls: print the state that would be sent
|
| Library use:
|     guard = CollisionGuard()
|     decision = guard.evaluate(scan, flight)       # scan/flight from LD06FlightClient.latest()
|     if decision.stop: ...
"""
import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DIRECTIONS = ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"]  # CCW, 45 deg


# --------------------------------------------------------------------------------------------- API key
def load_api_key(env_path=Path(__file__).with_name(".env")):
    """TYPESAFE_API_KEY from the environment, else from .env (either `TYPESAFE_API_KEY=...` or a bare key line)."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, _, value = line.partition("=")
                if name.strip().upper() in ("TYPESAFE_API_KEY", "API_KEY"):
                    return value.strip().strip("'\"")
            elif line.startswith("apikey_"):
                return line
    raise RuntimeError("No TypeSafe API key: set TYPESAFE_API_KEY or put it in .env")


# --------------------------------------------------------------------------------------------- state builder
def _proximity(d):
    if d is None:
        return "clear"
    if d < 1.0:
        return "very close"
    if d < 3.0:
        return "close"
    if d < 6.0:
        return "near"
    return "far"


def _urgency(ttc):
    if ttc is None:
        return "none"
    if ttc < 1.5:
        return "imminent"
    if ttc < 3.0:
        return "soon"
    if ttc < 6.0:
        return "later"
    return "none"


def analyze(scan, flight, moving_speed=0.3):
    """Pure-code geometry. Returns a dict with per-direction facts (numbers and words) and motion info."""
    s72 = scan["sectors_72"]  # 5 deg sectors, index 0 = nose, CCW, None = no return within 12 m

    # Body-frame velocity (FLU) from PX4 NED velocity and yaw
    vn, ve, yaw = flight.get("vn", 0.0), flight.get("ve", 0.0), flight.get("yaw", 0.0)
    v_fwd = vn * math.cos(yaw) + ve * math.sin(yaw)
    v_left = vn * math.sin(yaw) - ve * math.cos(yaw)
    speed = math.hypot(v_fwd, v_left)
    moving = speed >= moving_speed
    move_angle = math.atan2(v_left, v_fwd) if moving else None  # CCW from the nose

    dirs = []
    for k, name in enumerate(DIRECTIONS):
        c = k * 9
        vals = [s72[(c + j) % 72] for j in range(-4, 5) if s72[(c + j) % 72] is not None]
        dist = min(vals) if vals else None
        a = math.radians(k * 45)
        closing = v_fwd * math.cos(a) + v_left * math.sin(a)  # >0: moving toward this direction
        ttc = dist / closing if (dist is not None and closing > 0.1) else None
        in_path = moving and abs((a - move_angle + math.pi) % (2 * math.pi) - math.pi) <= math.radians(50)
        dirs.append({"direction": name, "distance_m": dist, "closing_speed_mps": closing,
                     "time_to_contact_s": ttc, "in_flight_path": in_path,
                     "proximity": _proximity(dist), "urgency": _urgency(ttc)})

    move_dir = None
    if moving:
        move_dir = DIRECTIONS[int(round(math.degrees(move_angle) / 45.0)) % 8]
    return {"speed": speed, "moving": moving, "moving_direction": move_dir, "directions": dirs,
            "min_distance": scan.get("min_distance")}


def build_state(scan, flight, facts):
    """The JSON state sent to Jev: worded facts, with rounded numbers kept only as supporting detail."""
    obstacles = []
    for d in facts["directions"]:
        if d["distance_m"] is None:
            continue
        o = {"direction": d["direction"], "proximity": d["proximity"], "distance": f"{d['distance_m']:.1f} m",
             "in_flight_path": "yes" if d["in_flight_path"] else "no"}
        if d["time_to_contact_s"] is not None:
            o["drone_movement"] = "approaching this obstacle"
            o["contact"] = f"{d['urgency']} (about {d['time_to_contact_s']:.1f} s)" if d["urgency"] != "none" \
                else "not soon"
        else:
            o["drone_movement"] = "not approaching this obstacle"
        obstacles.append(o)

    # Rank obstacles so the most dangerous comes first (lists are read in order)
    rank = {"imminent": 0, "soon": 1, "later": 2, "none": 3}
    obstacles.sort(key=lambda o: (rank.get(o.get("contact", "none").split(" ")[0], 3), float(o["distance"][:-2])))

    threats = [o for o in obstacles if o["in_flight_path"] == "yes" and o["drone_movement"].startswith("approaching")]
    if not facts["moving"]:
        summary = "The drone is hovering or nearly still."
    elif not threats:
        summary = f"The drone is moving {facts['moving_direction']} and no obstacle is in its flight path."
    else:
        t = threats[0]
        summary = (f"The drone is moving {facts['moving_direction']} toward a {t['proximity']} obstacle "
                   f"({t['distance']}) to the {t['direction']}; contact {t.get('contact', 'not soon')}.")

    return {
        "situation": "Quadcopter drone flying between buildings and moving traffic in a city. A 360-degree lidar "
                     "(range 12 m) reports obstacles around it. Decide whether the drone must stop to avoid a collision.",
        "summary": summary,
        "flight": {
            "armed": "yes" if flight.get("armed") else "no",
            "flight_mode": flight.get("mode") or "unknown",
            "altitude": f"{-flight['down']:.1f} m" if "down" in flight else "unknown",
            "speed": f"{facts['speed']:.1f} m/s",
            "moving": "yes" if facts["moving"] else "no",
            "moving_direction": facts["moving_direction"] or "none",
        },
        "obstacles": obstacles or "none within 12 m",
        "clear_directions": [d["direction"] for d in facts["directions"] if d["distance_m"] is None],
    }


# --------------------------------------------------------------------------------------------- Jev questions
def build_questions():
    from typesafe_sdk import Choice, Noul, NoulCriteria

    return {
        "stop_needed": Noul(
            instructions="The drone must stop now to avoid a collision.",
            criteria=NoulCriteria(
                true="An obstacle is in the drone's flight path and the drone is approaching it with contact "
                     "imminent or soon, or an obstacle is very close.",
                false="No obstacle is in the flight path, the drone is hovering, moving away, or obstacles are far.",
            ),
        ),
        "action": Choice(
            instructions="What should the drone do now?",
            criteria={
                "continue": "The flight path is clear; keep flying as commanded.",
                "slow_down": "An obstacle is in the flight path but contact is not imminent.",
                "stop": "Contact is imminent or an obstacle is very close in the flight path; brake and hold position.",
            },
        ),
        "threat_direction": Choice(
            instructions="In which direction is the most dangerous obstacle?",
            criteria={**{d: f"The most dangerous obstacle is to the {d}." for d in DIRECTIONS},
                      "none": "There is no dangerous obstacle."},
        ),
    }


# --------------------------------------------------------------------------------------------- decision
@dataclass
class StopDecision:
    stop: bool
    source: str                      # "rule", "jev", "rule+jev", "none", or "stale"
    reason: str
    stop_probability: float | None = None   # Jev noul for stop_needed
    action: str | None = None               # Jev recommended action
    action_confidence: float | None = None
    threat_direction: str | None = None
    rule_triggered: bool = False
    latency_s: float | None = None
    model: str | None = None
    state: dict = field(default_factory=dict)

    def to_text(self):
        jev = "" if self.stop_probability is None else (
            f" | jev p(stop)={self.stop_probability:.2f} action={self.action}({self.action_confidence:.2f}) "
            f"threat={self.threat_direction} {self.latency_s * 1000:.0f}ms")
        return f"{'STOP' if self.stop else 'ok  '} [{self.source}] {self.reason}{jev}"


class CollisionGuard:
    """
    Args:
        stop_threshold: Jev stop probability at/above which a stop is decided (docs: raise it when false stops
                        are costly, lower it when missing a stop is costly)
        action_confidence: minimum confidence for Jev's "stop" action to trigger a stop on its own
        hard_stop_distance: code rule, stop if an obstacle in the flight path is closer than this (m)
        hard_stop_ttc: code rule, stop if time to contact with an obstacle in the path is below this (s)
        min_clearance: code rule, stop if anything at all is closer than this (m)
        max_data_age: scans/telemetry older than this (s) are treated as stale -> stop
        model: TypeSafe model name
    """

    def __init__(self, stop_threshold=0.6, action_confidence=0.6, hard_stop_distance=1.5, hard_stop_ttc=1.0,
                 min_clearance=0.6, max_data_age=1.0, model="jev-latest", api_key=None, dry_run=False):
        self.stop_threshold = stop_threshold
        self.action_confidence = action_confidence
        self.hard_stop_distance = hard_stop_distance
        self.hard_stop_ttc = hard_stop_ttc
        self.min_clearance = min_clearance
        self.max_data_age = max_data_age
        self.model = model
        self.dry_run = dry_run
        self._client = None
        self._questions = None
        if not dry_run:
            from typesafe_sdk import RetryPolicy, TypeSafeClient
            self._client = TypeSafeClient(api_key=api_key or load_api_key(), model=model, timeout=5.0,
                                          retry=RetryPolicy(max_retries=1))
            self._questions = build_questions()

    def hard_rule(self, facts):
        """Deterministic stop rule, evaluated in code on every call. Returns a reason string or None."""
        for d in facts["directions"]:
            dist = d["distance_m"]
            if dist is None:
                continue
            if dist < self.min_clearance:
                return f"obstacle {dist:.2f} m to the {d['direction']} (< {self.min_clearance} m clearance)"
            if d["in_flight_path"] and dist < self.hard_stop_distance:
                return f"obstacle {dist:.2f} m in the flight path ({d['direction']})"
            if d["in_flight_path"] and d["time_to_contact_s"] is not None and d["time_to_contact_s"] < self.hard_stop_ttc:
                return f"contact in {d['time_to_contact_s']:.1f} s to the {d['direction']}"
        return None

    def evaluate(self, scan, flight, ask_jev=True):
        now = time.time()
        if scan is None or not flight or "down" not in flight:
            return StopDecision(True, "stale", "no lidar scan or flight telemetry yet")
        if now - scan.get("received", now) > self.max_data_age or now - flight.get("flight_received", now) > self.max_data_age:
            return StopDecision(True, "stale", "lidar or telemetry data is stale")

        facts = analyze(scan, flight)
        state = build_state(scan, flight, facts)
        rule = self.hard_rule(facts)

        if not ask_jev or self.dry_run:
            return StopDecision(rule is not None, "rule" if rule else "none", rule or "rule: path clear",
                                rule_triggered=rule is not None, state=state)

        t0 = time.time()
        resp = self._client.system_one(state=state, questions=self._questions)
        latency = time.time() - t0
        p_stop = resp.answers["stop_needed"].noul
        action = resp.answers["action"]
        threat = resp.answers["threat_direction"]

        jev_stop = p_stop >= self.stop_threshold or (action.choice == "stop" and action.confidence >= self.action_confidence)
        stop = rule is not None or jev_stop
        source = "rule+jev" if (rule and jev_stop) else "rule" if rule else "jev" if jev_stop else "none"
        reason = rule or (f"jev judged a collision risk ({state['summary']})" if jev_stop else state["summary"])
        return StopDecision(stop, source, reason, p_stop, action.choice, action.confidence, threat.choice,
                            rule is not None, latency, resp.model, state)


# --------------------------------------------------------------------------------------------- PX4 action
def command_hold(mav):
    """Switch PX4 to HOLD (AUTO.LOITER): the multicopter brakes and holds position."""
    from pymavlink import mavutil
    mav.mav.command_long_send(mav.target_system or 1, mav.target_component or 1,
                              mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                              mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                              4,   # PX4 main mode AUTO
                              3,   # PX4 sub mode LOITER (HOLD)
                              0, 0, 0, 0)


# --------------------------------------------------------------------------------------------- CLI loop
def main():
    ap = argparse.ArgumentParser(description="Jev collision-stop guard for the LD06 Iris")
    ap.add_argument("--act", action="store_true", help="send PX4 HOLD when a stop is decided (armed only)")
    ap.add_argument("--dry-run", action="store_true", help="no API calls; print the state that would be sent")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between Jev decisions")
    ap.add_argument("--threshold", type=float, default=0.6, help="Jev stop probability threshold")
    ap.add_argument("--scan-port", type=int, default=9006)
    ap.add_argument("--mavlink", default="udpin:0.0.0.0:14540")
    ap.add_argument("--show-state", action="store_true", help="print the JSON state sent to Jev")
    args = ap.parse_args()

    from ld06_flight_client import LD06FlightClient
    client = LD06FlightClient(scan_port=args.scan_port, mavlink_url=args.mavlink).start()
    guard = CollisionGuard(stop_threshold=args.threshold, dry_run=args.dry_run)
    print(f"Guard running ({'dry run' if args.dry_run else guard.model}); act={'on' if args.act else 'off'}")

    # Jev runs in its own thread so a slow API call never delays the code safety rule below
    latest = {"decision": None}

    def jev_worker():
        while True:
            scan, flight = client.latest()
            try:
                latest["decision"] = guard.evaluate(scan, flight)
            except Exception as e:  # network / API errors: the code rule keeps running
                print(f"[jev] error: {type(e).__name__}: {e}")
            time.sleep(args.interval)

    threading.Thread(target=jev_worker, daemon=True).start()

    last_hold = 0.0
    last_print = None
    while True:
        scan, flight = client.latest()
        fast = guard.evaluate(scan, flight, ask_jev=False)  # code rule at 10 Hz
        d = latest["decision"]
        stop = fast.stop and fast.source == "rule" or (d is not None and d.stop and d.source != "stale")
        shown = fast if fast.stop and fast.source == "rule" else (d or fast)
        line = shown.to_text()
        if line != last_print:
            print(time.strftime("%H:%M:%S"), line)
            if args.show_state:
                print(json.dumps(shown.state, indent=1))
            last_print = line
        if stop and args.act and flight.get("armed") and time.time() - last_hold > 2.0:
            command_hold(client.mav)
            last_hold = time.time()
            print(time.strftime("%H:%M:%S"), ">>> PX4 HOLD commanded")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
