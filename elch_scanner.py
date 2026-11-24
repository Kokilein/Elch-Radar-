import os
import re
import time
import json
import socket
from collections import Counter
import requests
import urllib3.util.connection as urllib3_cn
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

# Erzwingt IPv4, damit Telegram-Requests auf GitHub Actions nicht in IPv6-Rate-Limits laufen
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"].strip()
CHAT_ID = os.environ["CHAT_ID"].strip()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.vegvesen.no/"
}

FYLKER = {
    3: "Oslo", 11: "Rogaland", 15: "Møre og Romsdal", 18: "Nordland",
    30: "Viken", 31: "Østfold", 32: "Akershus", 33: "Buskerud",
    34: "Innlandet", 38: "Vestfold og Telemark", 39: "Vestfold",
    40: "Telemark", 42: "Agder", 46: "Vestland", 50: "Trøndelag",
    54: "Troms og Finnmark", 55: "Troms", 56: "Finnmark"
}

model = YOLO("yolov8s.pt")

ANIMAL_CLASSES = {17: "Pferd/Elch", 18: "Schaf/Rentier", 19: "Kuh/Elch", 21: "Bär/Elch"}
CONFIDENCE_THRESHOLD = 0.58
MAX_ALERTS_PER_RUN = 5

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def get_nvdb_cameras():
    cameras = []
    seen_ids = set()

    url = "https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/163?inkluder=egenskaper,lokasjon&antall=1000"
    nvdb_headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "application/vnd.vegvesen.nvdb-v4+json, application/json",
        "X-Client": "elch-radar-norwegen"
    }

    while url:
        try:
            res = requests.get(url, headers=nvdb_headers, timeout=25)
            if res.status_code != 200:
                print(f"NVDB-Fehler: Status {res.status_code}")
                break
            data = res.json()
            for obj in data.get("objekter", []):
                egenskaper = obj.get("egenskaper", [])
                msr_id = None
                direct_url = None
                place = None

                for eg in egenskaper:
                    eg_id = eg.get("id")
                    val = eg.get("verdi")
                    navn = str(eg.get("navn", "")).lower()

                    if (eg_id == 12211 or "måleutstyrsnummer" in navn) and val is not None:
                        msr_id = re.sub(r"\.0$", "", str(val).strip())
                    elif isinstance(val, str) and "kamera.atlas.vegvesen.no/api/images/" in val:
                        direct_url = val.strip()
                        if not msr_id:
                            msr_id = val.rstrip("/").split("/")[-1]
                    elif isinstance(val, str) and any(k in navn for k in ("stedsnavn", "navn", "beskrivelse", "plassering", "tittel")):
                        if not place and navn != "type kamera":
                            place = val.strip()

                if not msr_id or msr_id in seen_ids:
                    continue
                seen_ids.add(msr_id)

                lok = obj.get("lokasjon", {})
                vegsys = lok.get("vegsystemreferanser", [])
                road = vegsys[0].get("kortform", "").split(" ")[0] if vegsys else ""
                fylke_ids = lok.get("fylker", [])
                region = FYLKER.get(fylke_ids[0], "Norwegen") if fylke_ids else "Norwegen"

                title = f"{place or 'Kamera ' + msr_id} – {road}" if road else (place or f"Kamera {msr_id}")
                cameras.append({
                    "id": msr_id,
                    "name": f"{title} ({region})",
                    "direct_url": direct_url
                })

            next_url = data.get("metadata", {}).get("neste", {}).get("href")
            url = next_url if next_url and next_url != url else None
        except Exception as e:
            print(f"NVDB-Hinweis: {e}")
            break

    print(f"NVDB-Auswertung: {len(cameras)} Kameras mit MSR-Nummer gefunden.")
    return cameras


def try_fetch_url(url, cam_id, cam_name):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200 and len(r.content) > 5000 and not r.content[:20].lstrip().startswith(b"<"):
            img_hash = f"{cam_id}_{r.headers.get('Last-Modified', len(r.content))}"
            if img_hash in seen_images:
                return ("cached", None)
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", cam_id)
            filename = f"cam_{safe_id}.jpg"
            with open(filename, "wb") as f:
                f.write(r.content)
            return ("ok", {
                "id": cam_id,
                "name": cam_name,
                "url": url,
                "filename": filename,
                "img_hash": img_hash
            })
        return (str(r.status_code), None)
    except Exception as e:
        return (type(e).__name__, None)


def download_camera_images(cam):
    results = []
    msr_id = cam["id"]

    primary_url = cam["direct_url"] or f"https://kamera.atlas.vegvesen.no/api/images/{msr_id}"
    status, item = try_fetch_url(primary_url, msr_id, cam["name"])
    if status in ("ok", "cached"):
        return [(status, item)]

    if "_" not in msr_id:
        for suffix in ("1", "2", "3"):
            sub_id = f"{msr_id}_{suffix}"
            sub_url = f"https://kamera.atlas.vegvesen.no/api/images/{sub_id}"
            s_status, s_item = try_fetch_url(sub_url, sub_id, f"{cam['name']} [Blick {suffix}]")
            if s_status in ("ok", "cached"):
                results.append((s_status, s_item))

    return results if results else [(status, None)]


def send_telegram(image_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    for attempt in range(2):
        with open(image_path, "rb") as photo:
            res = requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": photo})
        if res.status_code == 200:
            return True
        if res.status_code == 429:
            wait = res.json().get("parameters", {}).get("retry_after", 10) + 1
            print(f"Telegram-Pause ({wait}s)...")
            time.sleep(wait)
        else:
            print(f"Telegram-Fehler: {res.status_code} {res.text}")
            return False
    return False


def main():
    cameras = get_nvdb_cameras()
    print(f"Prüfe {len(cameras)} Kameras in ganz Norwegen...")

    if not cameras:
        print("Abbruch: Keine Kameras gefunden.")
        return

    with ThreadPoolExecutor(max_workers=30) as executor:
        nested_results = list(executor.map(download_camera_images, cameras))

    flat_results = [entry for group in nested_results for entry in group]
    status_counts = Counter(status for status, _ in flat_results)
    print(f"Download-Statistik: {dict(status_counts)}")

    downloaded = [item for status, item in flat_results if status == "ok" and item]
    print(f"{len(downloaded)} neue Kamerabilder geladen. Starte Elch-Erkennung...")

    hits = []
    for item in downloaded:
        filename = item["filename"]
        try:
            results = model(filename, conf=CONFIDENCE_THRESHOLD, verbose=False)
            best_conf = 0.0
            best_label = None
            best_result = None

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id in ANIMAL_CLASSES and conf > best_conf:
                        best_conf = conf
                        best_label = ANIMAL_CLASSES[cls_id]
                        best_result = r

            if best_result is not None:
                safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", item["id"])
                out_file = f"hit_{safe_id}.jpg"
                best_result.save(filename=out_file)
                hits.append((best_conf, best_label, item, out_file))
        except Exception as e:
            print(f"Analysefehler bei {item['name']}: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    hits.sort(key=lambda x: x[0], reverse=True)
    print(f"Gefundene Treffer über {int(CONFIDENCE_THRESHOLD * 100)}%: {len(hits)}")

    for conf, label, item, out_file in hits[:MAX_ALERTS_PER_RUN]:
        caption = (
            f"🦌 Mögliche Sichtung ({label} – {int(conf * 100)}%)\n"
            f"📍 Ort: {item['name']}\n"
            f"🔗 Live-Bild: {item['url']}"
        )
        if send_telegram(out_file, caption):
            seen_images.add(item["img_hash"])
            print(f"TREFFER gemeldet: {item['name']} ({label} {int(conf * 100)}%)")
            time.sleep(1.5)

    for _, _, _, out_file in hits:
        if os.path.exists(out_file):
            os.remove(out_file)

    with open(HISTORY_FILE, "w") as f:
        json.dump(list(seen_images)[-2000:], f)


if __name__ == "__main__":
    main()
