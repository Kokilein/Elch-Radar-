import os
import re
import json
from collections import Counter
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

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

model = YOLO("yolov8n.pt")
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def build_image_urls(raw_id, direct_url=None):
    urls = []
    if direct_url and "kamera.atlas.vegvesen.no/api/images/" in direct_url:
        urls.append(direct_url)
    clean_id = str(raw_id).strip()
    if clean_id:
        for candidate in (clean_id, f"{clean_id}_1", f"{clean_id}_2", f"{clean_id}_3"):
            if "_" in clean_id and candidate != clean_id:
                continue
            u = f"https://kamera.atlas.vegvesen.no/api/images/{candidate}"
            if u not in urls:
                urls.append(u)
    return urls


def get_nvdb_cameras():
    cameras = []
    seen_ids = set()
    skipped_no_id = 0
    sample_without_id = []

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
                ext_id = None
                direct_url = None
                place = None

                for eg in egenskaper:
                    val = eg.get("verdi")
                    navn = str(eg.get("navn", "")).lower()

                    if isinstance(val, str) and "http" in val:
                        direct_url = val
                        if "/api/images/" in val:
                            ext_id = val.rstrip("/").split("/")[-1]
                        elif "id=" in val:
                            ext_id = val.split("id=")[-1].split("&")[0]
                    elif val and any(k in navn for k in ("webkamera", "kameraid", "kamera-id", "kameranummer", "ekstern")):
                        if not ext_id:
                            ext_id = str(val).strip()
                    elif isinstance(val, str) and any(k in navn for k in ("stedsnavn", "navn", "beskrivelse", "plassering")):
                        if not place:
                            place = val

                if not ext_id:
                    skipped_no_id += 1
                    if len(sample_without_id) < 2:
                        sample_without_id.append(egenskaper)
                    continue

                if ext_id in seen_ids:
                    continue
                seen_ids.add(ext_id)

                lok = obj.get("lokasjon", {})
                vegsys = lok.get("vegsystemreferanser", [])
                road = vegsys[0].get("kortform", "").split(" ")[0] if vegsys else ""
                fylke_ids = lok.get("fylker", [])
                region = FYLKER.get(fylke_ids[0], "Norwegen") if fylke_ids else "Norwegen"

                title = f"{place or 'Kamera ' + ext_id} – {road}" if road else (place or f"Kamera {ext_id}")
                cameras.append({
                    "id": ext_id,
                    "name": f"{title} ({region})",
                    "urls": build_image_urls(ext_id, direct_url),
                    "raw_egenskaper": egenskaper
                })

            next_url = data.get("metadata", {}).get("neste", {}).get("href")
            url = next_url if next_url and next_url != url else None
        except Exception as e:
            print(f"NVDB-Hinweis: {e}")
            break

    print(f"NVDB-Auswertung: {len(cameras)} Kameras mit Web-ID gefunden, {skipped_no_id} Einträge ohne Web-ID übersprungen.")
    if sample_without_id:
        print("DIAGNOSE (Beispiel ohne Web-ID):", json.dumps(sample_without_id[0], ensure_ascii=False)[:500])

    return cameras


def download_image(cam):
    last_status = "error"
    for url in cam["urls"]:
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            last_status = str(r.status_code)
            if r.status_code == 200 and len(r.content) > 5000 and not r.content[:20].lstrip().startswith(b"<"):
                img_hash = f"{cam['id']}_{r.headers.get('Last-Modified', len(r.content))}"
                if img_hash in seen_images:
                    return ("cached", None)
                safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", cam["id"])
                filename = f"cam_{safe_id}.jpg"
                with open(filename, "wb") as f:
                    f.write(r.content)
                cam["active_url"] = url
                return ("ok", (cam, filename, img_hash))
        except Exception as e:
            last_status = type(e).__name__
    return (last_status, cam)


def send_telegram(image_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": photo})


def main():
    cameras = get_nvdb_cameras()
    print(f"Prüfe {len(cameras)} Kameras in ganz Norwegen...")

    if not cameras:
        print("Abbruch: Keine Kameras gefunden.")
        return

    with ThreadPoolExecutor(max_workers=25) as executor:
        results_raw = list(executor.map(download_image, cameras))

    status_counts = Counter(status for status, _ in results_raw)
    print(f"Download-Statistik: {dict(status_counts)}")

    downloaded = [payload for status, payload in results_raw if status == "ok" and payload]
    failed_cams = [payload for status, payload in results_raw if status not in ("ok", "cached") and payload]

    if failed_cams:
        print("DIAGNOSE (Beispiel fehlgeschlagene Kamera):", failed_cams[0]["name"], "->", json.dumps(failed_cams[0]["raw_egenskaper"], ensure_ascii=False)[:500])

    print(f"{len(downloaded)} neue Kamerabilder geladen. Starte Elch-Erkennung...")

    for cam, filename, img_hash in downloaded:
        try:
            results = model(filename, conf=0.45, verbose=False)
            for r in results:
                detected = [int(box.cls[0]) for box in r.boxes]
                if any(cls in ANIMAL_CLASSES for cls in detected):
                    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", cam["id"])
                    out_file = f"hit_{safe_id}.jpg"
                    r.save(filename=out_file)
                    caption = (
                        f"🦌 Mögliche Elch-Sichtung!\n"
                        f"📍 Ort: {cam['name']}\n"
                        f"🔗 Live-Bild: {cam.get('active_url', cam['urls'][0])}"
                    )
                    send_telegram(out_file, caption)
                    seen_images.add(img_hash)
                    print(f"TREFFER gemeldet: {cam['name']}")
                    if os.path.exists(out_file):
                        os.remove(out_file)
                    break
        except Exception as e:
            print(f"Analysefehler bei {cam['name']}: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    with open(HISTORY_FILE, "w") as f:
        json.dump(list(seen_images)[-2000:], f)


if __name__ == "__main__":
    main()
