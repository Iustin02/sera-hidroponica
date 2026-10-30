#!/usr/bin/env python3
"""
Baza de date SQLite a sistemului NFT (teza, Cap. 4.2, Anexele A si G).

init_db()        creeaza schema (plants, sensor_readings) cu constrangeri CHECK
insert_reading() inregistreaza o citire a celor sapte senzori
latest_reading() ultima citire, folosita de bucla de control

Rulare directa: creeaza baza si insereaza o citire exemplu.
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "date" / "sera_hidroponica.db"

SENSOR_COLUMNS = ("ph", "ec_ms_cm", "water_temp_c", "ambient_temp_c",
                  "humidity_pct", "light_lux", "water_level_pct")

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS plants (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    species    TEXT,
    notes      TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id        INTEGER,
    reading_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ph              REAL CHECK (ph IS NULL OR (ph BETWEEN 0.0 AND 14.0)),
    ec_ms_cm        REAL CHECK (ec_ms_cm IS NULL OR (ec_ms_cm BETWEEN 0.0 AND 10.0)),
    water_temp_c    REAL CHECK (water_temp_c IS NULL OR (water_temp_c BETWEEN 0.0 AND 40.0)),
    ambient_temp_c  REAL CHECK (ambient_temp_c IS NULL OR (ambient_temp_c BETWEEN -10.0 AND 50.0)),
    humidity_pct    REAL CHECK (humidity_pct IS NULL OR (humidity_pct BETWEEN 0.0 AND 100.0)),
    light_lux       REAL CHECK (light_lux IS NULL OR (light_lux BETWEEN 0.0 AND 100000.0)),
    water_level_pct REAL CHECK (water_level_pct IS NULL OR (water_level_pct BETWEEN 0.0 AND 100.0)),
    notes           TEXT,
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_reading_at ON sensor_readings(reading_at);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_plant_id   ON sensor_readings(plant_id);
"""


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Creeaza schema completa a bazei de date cu constrangeri CHECK."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_reading(conn: sqlite3.Connection, plant_id: int | None, **values: float) -> None:
    """Insereaza o citire de senzori. Toate campurile sunt optionale."""
    unknown = set(values) - set(SENSOR_COLUMNS)
    if unknown:
        raise ValueError(f"Coloane necunoscute: {sorted(unknown)}")
    cols = ", ".join(["plant_id", *values])
    marks = ", ".join("?" * (len(values) + 1))
    conn.execute(f"INSERT INTO sensor_readings ({cols}) VALUES ({marks})", [plant_id, *values.values()])
    conn.commit()


def latest_reading(conn: sqlite3.Connection) -> dict:
    """Returneaza ultima citire ca dict {coloana: valoare} (gol daca nu exista)."""
    row = conn.execute(
        f"SELECT {', '.join(SENSOR_COLUMNS)} FROM sensor_readings ORDER BY reading_at DESC, id DESC LIMIT 1"
    ).fetchone()
    return dict(zip(SENSOR_COLUMNS, row)) if row else {}


def ensure_plant(conn: sqlite3.Connection, name: str = "Busuioc NFT", species: str = "Ocimum basilicum") -> int:
    conn.execute("INSERT OR IGNORE INTO plants (name, species) VALUES (?, ?)", (name, species))
    conn.commit()
    return conn.execute("SELECT id FROM plants WHERE name = ?", (name,)).fetchone()[0]


if __name__ == "__main__":
    conn = init_db()
    plant_id = ensure_plant(conn)
    insert_reading(
        conn, plant_id,
        ph=6.1,                # pH-4502C: zona optima 5.5-6.5
        ec_ms_cm=1.8,          # EC: zona optima busuioc 1.2-2.4 mS/cm
        water_temp_c=22.5,     # DS18B20: zona optima 18-24 C
        ambient_temp_c=24.0,   # DHT22: temperatura aerului
        humidity_pct=68.0,     # DHT22: umiditate relativa
        light_lux=12500.0,     # BH1750
        water_level_pct=75.0,  # senzor nivel bazin
    )
    print(f"Citire salvata in {DB_PATH}: {latest_reading(conn)}")
