#!/usr/bin/env python3
"""
Inferenta YOLOv8s-cls in timp real pe Raspberry Pi 5 (teza, Anexa E).

Captureaza cadre de la camera USB (/dev/video0, OpenCV V4L2), clasifica starea
fitosanitara si mapeaza rezultatul la o decizie de control.

Erata fata de Anexa E: indicii claselor se citesc din model (`model.names`).
Ultralytics ordoneaza clasele alfabetic dupa folder -> 0=fusarium, 1=healthy,
2=powdery; maparea fixa {0: healthy, ...} din anexa era gresita.

Utilizare:
  python src/edge/inference_rpi5.py --weights models/basil_health_c5.pt --imgsz 416
  python src/edge/inference_rpi5.py --source frunza.jpg
"""

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LABELS = {
    "healthy": "healthy",
    "fusarium": "Fusarium oxysporum",
    "powdery": "Oidium belbahrii",
}
ALERT_CLASSES = {"fusarium", "powdery"}  # clasele care declanseaza alerta
CONF_THRESHOLD = 0.70                    # ignora predictii sub 70% confidence


def load_model(weights: str):
    from ultralytics import YOLO

    model = YOLO(weights)
    model.fuse()  # fuse Conv+BN: inferenta mai rapida
    return model


def classify_frame(model, frame, imgsz: int) -> dict:
    """Ruleaza inferenta pe un cadru BGR (OpenCV) sau pe o cale de imagine."""
    result = model.predict(source=frame, imgsz=imgsz, verbose=False, device="cpu")[0]
    key = model.names[int(result.probs.top1)]
    conf = float(result.probs.top1conf)
    return {
        "class_key": key,
        "class_name": LABELS.get(key, key),
        "confidence": conf,
        "alert": key in ALERT_CLASSES and conf >= CONF_THRESHOLD,
    }


def decide_action(pred: dict) -> str:
    """Mapeaza clasa detectata la recomandarea de control."""
    if not pred["alert"]:
        return "NORMAL — nicio actiune"
    if pred["class_key"] == "fusarium":
        return "ALERTA Fuzarioza: verifica pH (5.5-6.5) si EC (1.6-2.2 mS/cm)"
    if pred["class_key"] == "powdery":
        return "ALERTA Fainare: creste ventilatia, scade umiditatea sub 60%"
    return "ALERTA: inspectie manuala recomandata"


def report(pred: dict, latency_ms: float) -> None:
    print(f"[{latency_ms:6.1f} ms] [{pred['confidence']:.0%}] "
          f"{pred['class_name']:20s} -> {decide_action(pred)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(ROOT / "models" / "basil_health_c3.pt"))
    parser.add_argument("--imgsz", type=int, default=320, help="320 pentru C3, 416 pentru C5")
    parser.add_argument("--source", help="imagine; implicit camera USB 0")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.weights)

    if args.source:
        t0 = time.perf_counter()
        pred = classify_frame(model, args.source, args.imgsz)
        report(pred, (time.perf_counter() - t0) * 1000)
        return

    import cv2

    cap = cv2.VideoCapture(args.camera)
    print("Inferenta pornita. Ctrl+C pentru oprire.")
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            pred = classify_frame(model, frame, args.imgsz)
            report(pred, (time.perf_counter() - t0) * 1000)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
