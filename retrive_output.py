import time
from ld06_flight_client import LD06FlightClient

DIRS = ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"]

def to_text(scan, fl):
    lines = []
    if fl:
        lines.append(
            f"FLIGHT: mode={fl.get('mode')} armed={fl.get('armed')} "
            f"pos_north={fl.get('north', 0):.1f}m pos_east={fl.get('east', 0):.1f}m "
            f"alt={-fl.get('down', 0):.1f}m heading={fl.get('heading_deg')}deg "
            f"speed={fl.get('groundspeed', 0):.1f}m/s battery={fl.get('battery_pct')}%")
    if scan:
        s = scan["sectors_72"]                       # 72 x 5 deg, index 0 = nose, counter-clockwise
        parts = []
        for k, name in enumerate(DIRS):              # 8 directions, 45 deg wide each
            c = k * 9
            vals = [s[(c + j) % 72] for j in range(-4, 5) if s[(c + j) % 72] is not None]
            parts.append(f"{name}={min(vals):.1f}m" if vals else f"{name}=clear")
        lines.append("LIDAR (max 12m): " + ", ".join(parts))
    return "\n".join(lines)

client = LD06FlightClient().start()
while True:
    scan, flight = client.latest()
    text = to_text(scan, flight)
    print(text)              # replace with: action = your_model(text)
    time.sleep(0.2)
