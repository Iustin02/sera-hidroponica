#!/usr/bin/env python3
"""
Control in bucla inchisa cu histerezis pentru sistemul NFT (teza, Anexa I).

La fiecare ciclu citeste ultima inregistrare din SQLite si actioneaza pompele
peristaltice de dozare si ventilatorul.

Implicit ruleaza in mod simulare (--dry-run): comenzile sunt doar afisate.
Cu --hardware foloseste gpiozero pe pinii validati pe banc (vezi docs/hardware.md).

Erata fata de Anexa I: interogarea SQL foloseste acum numele reale ale coloanelor
din schema (ec_ms_cm, ambient_temp_c, reading_at), nu ec/temp_air/timestamp.
"""

import argparse
import time
from dataclasses import dataclass

from database import DB_PATH, init_db, latest_reading


@dataclass
class Setpoints:
    ph_min: float = 5.8
    ph_max: float = 6.5
    ec_min: float = 1.6          # mS/cm
    ec_max: float = 2.4
    temp_air_max: float = 28.0   # C — peste prag: ventilatie (stres termic)
    humidity_max: float = 70.0   # % — peste prag: ventilatie (risc Oidium)


SP = Setpoints()
DOSE_AB_S = 2.0          # durata dozare nutrient A / B
LOOP_INTERVAL_S = 30

# Pini BCM validati pe banc (2026-06-30)
PIN_DOSE_A = (5, 6)      # pompa peristaltica 1 — punte H, intrari 1A/1B
PIN_DOSE_B = (13, 19)    # pompa peristaltica 2 — punte H, intrari 2A/2B
PIN_FAN_RELAY = 16       # releu ventilator (on/off)


class Actuators:
    """Abstractie peste GPIO; in dry-run doar afiseaza comenzile."""

    def __init__(self, hardware: bool):
        self.hardware = hardware
        if hardware:
            from gpiozero import Motor, OutputDevice

            self.pumps = {"NUTRIENT_A": Motor(*PIN_DOSE_A), "NUTRIENT_B": Motor(*PIN_DOSE_B)}
            self.fan = OutputDevice(PIN_FAN_RELAY)

    def dose(self, pump: str, seconds: float) -> None:
        print(f"  [POMPA {pump}] ON {seconds:.1f}s")
        try:
            if self.hardware:
                self.pumps[pump].forward()
            time.sleep(seconds)
        finally:  # pompa se opreste si la Ctrl+C sau exceptie
            if self.hardware:
                self.pumps[pump].stop()
            print(f"  [POMPA {pump}] OFF")

    def set_fan(self, on: bool) -> None:
        print(f"  [VENTILATOR] {'ON' if on else 'OFF'}")
        if self.hardware:
            self.fan.on() if on else self.fan.off()


def control_cycle(sensors: dict, act: Actuators) -> None:
    ph, ec = sensors.get("ph"), sensors.get("ec_ms_cm")
    t_air, rh = sensors.get("ambient_temp_c"), sensors.get("humidity_pct")

    if ph is not None and not SP.ph_min <= ph <= SP.ph_max:
        print(f"  AVERTISMENT pH={ph:.2f} in afara [{SP.ph_min}, {SP.ph_max}] — corectie manuala")
    if ec is not None and ec < SP.ec_min:
        act.dose("NUTRIENT_A", DOSE_AB_S)
        act.dose("NUTRIENT_B", DOSE_AB_S)
    elif ec is not None and ec > SP.ec_max:
        print(f"  AVERTISMENT EC={ec:.2f} mS/cm peste {SP.ec_max} — diluare necesara")

    hot = t_air is not None and t_air > SP.temp_air_max
    humid = rh is not None and rh > SP.humidity_max
    act.set_fan(hot or humid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="simulare (implicit)")
    mode.add_argument("--hardware", action="store_true", help="comanda GPIO reala (doar pe Pi)")
    parser.add_argument("--once", action="store_true", help="un singur ciclu")
    args = parser.parse_args()

    conn = init_db(DB_PATH)
    act = Actuators(hardware=args.hardware)
    print(f"Control NFT pornit ({'hardware' if args.hardware else 'dry-run'}), interval {LOOP_INTERVAL_S}s")
    while True:
        sensors = latest_reading(conn)
        if sensors:
            control_cycle(sensors, act)
        else:
            print("  AVERTISMENT: lipsa date senzori")
        if args.once:
            break
        time.sleep(LOOP_INTERVAL_S)


if __name__ == "__main__":
    main()
