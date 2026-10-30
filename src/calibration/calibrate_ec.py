#!/usr/bin/env python3
"""
Calibrare in punct unic a senzorului EC/TDS cu solutie etalon 1413 uS/cm
(teza, Cap. 3.2.2, Anexa K).

Factorul K = 1413 / EC_masurat (compensat termic, 2%/C fata de 25 C) se aplica
tuturor masuratorilor ulterioare.

Rezultat: config/ec_calibration.json
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIB_FILE = ROOT / "config" / "ec_calibration.json"
EC_REFERENCE_US = 1413.0
TEMP_REFERENCE_C = 25.0
TEMP_COEFF = 0.02

try:
    import adafruit_ads1x15.ads1115 as ADS
    import board
    import busio
    from adafruit_ads1x15.analog_in import AnalogIn

    _chan = AnalogIn(ADS.ADS1115(busio.I2C(board.SCL, board.SDA)), ADS.P1)

    def read_raw_ec() -> float:
        return _chan.voltage * 1000.0
except Exception:
    def read_raw_ec() -> float:
        return float(input("  Valoare EC bruta (uS/cm): "))


def read_stable(samples: int = 15, delay: float = 0.3) -> float:
    vals = []
    for _ in range(samples):
        vals.append(read_raw_ec())
        time.sleep(delay)
    return sum(vals) / len(vals)


def compensate(ec_raw: float, temp_c: float) -> float:
    return ec_raw / (1 + TEMP_COEFF * (temp_c - TEMP_REFERENCE_C))


def calibrate(temp_c: float = 25.0) -> dict:
    input("  Imerseaza electrodul in 1413 uS/cm. Enter cand e stabil (~30 s)...")
    raw = read_stable()
    raw_c = compensate(raw, temp_c)
    k = EC_REFERENCE_US / raw_c if raw_c else 1.0
    calib = {"k_factor": k, "ec_reference_us": EC_REFERENCE_US,
             "raw_measured": raw, "temp_calibration_c": temp_c}
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps(calib, indent=2))
    print(f"  K={k:.4f}  salvat: {CALIB_FILE}")
    return calib


def raw_to_ec(raw: float, temp_c: float, calib: dict) -> float:
    return compensate(raw, temp_c) * calib["k_factor"]


if __name__ == "__main__":
    temp = float(input("Temperatura solutiei (C) [25.0]: ") or "25.0")
    calib = calibrate(temp_c=temp)
    while True:
        raw = read_stable(samples=5)
        ec = raw_to_ec(raw, temp, calib)
        print(f"  raw={raw:.1f}  EC={ec:.1f} uS/cm  ({ec / 1000:.3f} mS/cm)")
        time.sleep(2)
