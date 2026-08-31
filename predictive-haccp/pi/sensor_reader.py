"""
pi/sensor_reader.py
---------------------
Reads temperature, humidity (DHT22) and door state (reed switch) from
two refrigerators and logs readings to a CSV file.

Hardware:
  - Raspberry Pi 4
  - 2× DHT22 sensors  (one per fridge)
  - 2× Reed switches  (one per fridge door)

Simulated mode:
  If 'adafruit_dht' or 'gpiozero' are not available (e.g. running on a
  laptop for testing), the script falls back to SIMULATED readings so the
  rest of the pipeline can be tested without Pi hardware.

Usage (on Pi):
    python3 pi/sensor_reader.py

Usage (simulated, any machine):
    python3 pi/sensor_reader.py --simulate
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).parent / "config.example.json"

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

LOG_CFG     = CONFIG["logging"]
CSV_PATH    = Path(__file__).parent.parent / LOG_CFG["csv_path"]
INTERVAL    = LOG_CFG["interval_seconds"]
ROLL_WINDOW = LOG_CFG["rolling_window"]

FRIDGES = [CONFIG["fridge_a"], CONFIG["fridge_b"]]

CSV_HEADER = [
    "timestamp", "fridge_id",
    "temperature", "humidity",
    "door_state", "door_open_count", "door_open_duration_s",
    "temp_rolling_mean", "temp_rolling_std",
    "humidity_rolling_mean", "humidity_rolling_std",
    "temp_delta", "humidity_delta",
    "hour_of_day",
    "sensor_ok",
]

# ── Rolling buffers per fridge ────────────────────────────────────────────────
_temp_buffers  = {f["fridge_id"]: deque(maxlen=ROLL_WINDOW) for f in FRIDGES}
_hum_buffers   = {f["fridge_id"]: deque(maxlen=ROLL_WINDOW) for f in FRIDGES}
_door_events   = {f["fridge_id"]: [] for f in FRIDGES}          # list of (open_ts, close_ts)
_door_open_ts  = {f["fridge_id"]: None for f in FRIDGES}       # time door opened
_prev_temp     = {f["fridge_id"]: None for f in FRIDGES}
_prev_hum      = {f["fridge_id"]: None for f in FRIDGES}


# ── Hardware initialisation ───────────────────────────────────────────────────
def init_hardware():
    """Initialise DHT22 sensors and reed-switch GPIO pins. Returns sensor objects."""
    try:
        import board
        import adafruit_dht
        from gpiozero import Button

        sensors = {}
        doors   = {}
        for fridge in FRIDGES:
            fid      = fridge["fridge_id"]
            dht_pin  = getattr(board, f"D{fridge['dht_pin']}")
            door_pin = fridge["door_pin"]
            sensors[fid] = adafruit_dht.DHT22(dht_pin, use_pulseio=False)
            doors[fid]   = Button(door_pin, pull_up=True)
            log.info(f"Initialised {fid}: DHT22 on GPIO{fridge['dht_pin']}, "
                     f"reed switch on GPIO{door_pin}")
        return sensors, doors, False   # False = not simulated

    except (ImportError, AttributeError, RuntimeError) as e:
        log.warning(f"Hardware not available ({e}). Switching to SIMULATED mode.")
        return {}, {}, True


def read_dht22_simulated(fridge_id: str):
    """Return plausible simulated temperature and humidity."""
    base_temp = 4.0 if "a" in fridge_id else 5.5
    base_hum  = 85.0 if "a" in fridge_id else 78.0
    temp = round(base_temp + random.gauss(0, 0.3), 1)
    hum  = round(min(100.0, max(50.0, base_hum + random.gauss(0, 1.5))), 1)
    return temp, hum


def read_dht22(sensor, fridge_id: str, simulated: bool):
    """Read one DHT22. Returns (temp, humidity, ok_flag)."""
    if simulated:
        t, h = read_dht22_simulated(fridge_id)
        return t, h, True
    try:
        return sensor.temperature, sensor.humidity, True
    except RuntimeError as e:
        log.warning(f"{fridge_id} DHT22 read error: {e}")
        return None, None, False


def read_door(door_button, fridge_id: str, simulated: bool) -> int:
    """Return 1 if door is open, 0 if closed."""
    if simulated:
        # Simulate fridge_b opening more frequently
        prob = 0.05 if "a" in fridge_id else 0.15
        return 1 if random.random() < prob else 0
    return 1 if door_button.is_pressed else 0


def rolling_stats(buf: deque):
    """Return (mean, std) or (None, None) if buffer is empty."""
    if not buf:
        return None, None
    arr = list(buf)
    n   = len(arr)
    mean = sum(arr) / n
    std  = math.sqrt(sum((x - mean) ** 2 for x in arr) / n) if n > 1 else 0.0
    return round(mean, 3), round(std, 3)


def update_door_events(fridge_id: str, door_state: int, now: datetime):
    """Track door open/close events within the rolling window."""
    if door_state == 1 and _door_open_ts[fridge_id] is None:
        _door_open_ts[fridge_id] = now
    elif door_state == 0 and _door_open_ts[fridge_id] is not None:
        open_ts  = _door_open_ts[fridge_id]
        duration = (now - open_ts).total_seconds()
        _door_events[fridge_id].append((open_ts, now, duration))
        _door_open_ts[fridge_id] = None

    # Prune events older than rolling window (seconds)
    cutoff = (datetime.now() - __import__("datetime").timedelta(
        seconds=INTERVAL * ROLL_WINDOW)).timestamp()
    _door_events[fridge_id] = [
        e for e in _door_events[fridge_id]
        if e[1].timestamp() > cutoff
    ]


def get_door_stats(fridge_id: str, door_state: int, now: datetime):
    update_door_events(fridge_id, door_state, now)
    events   = _door_events[fridge_id]
    count    = len(events)
    duration = sum(e[2] for e in events)
    # Add current open duration if door is still open
    if _door_open_ts[fridge_id] is not None:
        duration += (now - _door_open_ts[fridge_id]).total_seconds()
    return count, round(duration, 1)


def collect_reading(fridge: dict, sensor, door_button,
                    simulated: bool) -> dict | None:
    """Collect one reading row for a single fridge."""
    fid  = fridge["fridge_id"]
    now  = datetime.now()
    hour = now.hour

    temp, hum, ok = read_dht22(sensor, fid, simulated)
    door_state     = read_door(door_button, fid, simulated)

    if temp is None or hum is None:
        log.warning(f"{fid}: skipping row — bad sensor reading")
        return None

    # Update rolling buffers
    _temp_buffers[fid].append(temp)
    _hum_buffers[fid].append(hum)

    temp_mean, temp_std = rolling_stats(_temp_buffers[fid])
    hum_mean,  hum_std  = rolling_stats(_hum_buffers[fid])

    temp_delta = round(temp - _prev_temp[fid], 3) if _prev_temp[fid] is not None else 0.0
    hum_delta  = round(hum  - _prev_hum[fid],  3) if _prev_hum[fid]  is not None else 0.0

    _prev_temp[fid] = temp
    _prev_hum[fid]  = hum

    door_count, door_dur = get_door_stats(fid, door_state, now)

    return {
        "timestamp":            now.strftime("%Y-%m-%d %H:%M:%S"),
        "fridge_id":            fid,
        "temperature":          temp,
        "humidity":             hum,
        "door_state":           door_state,
        "door_open_count":      door_count,
        "door_open_duration_s": door_dur,
        "temp_rolling_mean":    temp_mean,
        "temp_rolling_std":     temp_std,
        "humidity_rolling_mean":hum_mean,
        "humidity_rolling_std": hum_std,
        "temp_delta":           temp_delta,
        "humidity_delta":       hum_delta,
        "hour_of_day":          hour,
        "sensor_ok":            int(ok),
    }


def ensure_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()
        log.info(f"Created CSV log: {path}")


def append_row(path: Path, row: dict):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)


def main(simulate: bool = False):
    log.info("=== Predictive HACCP — Raspberry Pi Sensor Reader ===")
    log.info(f"Mode: {'SIMULATED' if simulate else 'HARDWARE'}")
    log.info(f"Interval: {INTERVAL}s | Rolling window: {ROLL_WINDOW} readings")
    log.info(f"CSV log: {CSV_PATH}")

    sensors, doors, hw_sim = init_hardware()
    use_sim = simulate or hw_sim

    ensure_csv(CSV_PATH)

    try:
        while True:
            for fridge in FRIDGES:
                fid    = fridge["fridge_id"]
                sensor = sensors.get(fid)
                door   = doors.get(fid)
                row    = collect_reading(fridge, sensor, door, use_sim)
                if row:
                    append_row(CSV_PATH, row)
                    log.info(
                        f"{fid} | T={row['temperature']}°C "
                        f"H={row['humidity']}% "
                        f"Door={row['door_state']} "
                        f"DoorCount={row['door_open_count']}"
                    )
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        log.info("Sensor reader stopped by user.")
        for s in sensors.values():
            try:
                s.exit()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="Run in simulated mode (no Pi hardware needed)")
    args = parser.parse_args()
    main(simulate=args.simulate)
