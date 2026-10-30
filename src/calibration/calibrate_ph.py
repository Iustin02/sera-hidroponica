#!/usr/bin/env python3
"""
Calibrare in 3 puncte a senzorului pH-4502C (teza, Cap. 3.2.1, Anexa J).

Citeste tensiunea ADS1115 (canal A0) in solutiile tampon pH 4,01 / 6,86 / 9,18
si calculeaza regresie liniara pe doua segmente. Fara hardware, tensiunea se
introduce manual (util pentru test pe laptop).

Rezultat: config/ph_calibration.json
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIB_FILE = ROOT / "config" / "ph_calibration.json"
BUFFERS = [4.01, 6.86, 9.18]

try:
    import adafruit_ads1x15.ads1115 as ADS
    import board
    import busio
    from adafruit_ads1x15.analog_in import AnalogIn

    _chan = AnalogIn(ADS.ADS1115(busio.I2C(board.SCL, board.SDA)), ADS.P0)

    def read_voltage() -> float:
        return _chan.voltage
except Exception:
    def read_voltage() -> float:
        return float(input("  Tensiune ADC (V): "))


def read_stable(samples: int = 10, delay: float = 0.2) -> float:
    readings = []
    for _ in range(samples):
        readings.append(read_voltage())
        time.sleep(delay)
    return sum(readings) / len(readings)


def segment(ph1: float, v1: float, ph2: float, v2: float) -> dict:
    slope = (ph2 - ph1) / (v2 - v1) if v2 != v1 else 0.0
    return {"slope": slope, "intercept": ph1 - slope * v1, "v_ref": v1}


def calibrate() -> dict:
    print("=== Calibrare pH-4502C — 3 puncte ===")
    voltages = []
    for ph_ref in BUFFERS:
        input(f"  Imerseaza sonda in pH {ph_ref:.2f}. Enter cand e stabil...")
        v = read_stable()
        voltages.append(v)
        print(f"  V={v:.4f} -> pH {ph_ref:.2f}")
    calib = {
        "seg1": segment(BUFFERS[0], voltages[0], BUFFERS[1], voltages[1]),
        "seg2": segment(BUFFERS[1], voltages[1], BUFFERS[2], voltages[2]),
        "v_mid": voltages[1], "buffers": BUFFERS, "voltages": voltages,
    }
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps(calib, indent=2))
    print(f"Calibrare salvata: {CALIB_FILE}")
    return calib


def voltage_to_ph(v: float, calib: dict) -> float:
    # Segmentul acid (seg1) e de aceeasi parte a lui v_mid ca tensiunea tamponului pH 4,01.
    # Erata Anexa J: anexa presupunea panta negativa; modulul folosit are panta pozitiva.
    acid_side_high = calib["voltages"][0] >= calib["v_mid"]
    in_acid_zone = (v >= calib["v_mid"]) == acid_side_high
    seg = calib["seg1"] if in_acid_zone else calib["seg2"]
    return seg["slope"] * v + seg["intercept"]


if __name__ == "__main__":
    calib = calibrate()
    while True:
        v = read_stable(samples=5)
        print(f"  V={v:.4f}  pH={voltage_to_ph(v, calib):.2f}")
        time.sleep(2)
