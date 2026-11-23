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

# Eigenschaften, die keine Kamera-ID sind (z. B. Baujahr)
IGNORE_ID_KEYWORDS = ("år", "year", "høyde", "bredde", "lengde", "retning", "avstand")

model = YOLO("yolov8n.pt")
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def build_image_urls(candidate_ids, direct_urls):
    urls = []
    for u in direct_urls:
        if "kamera.atlas.vegvesen.no/api/images/" in u and u not in urls:
            urls.append(u)
    for raw_id in candidate_ids:
        clean_id = str(raw_id).strip()
        if not clean_id:
            continue
        variants = [clean_id] if "_" in clean_id else [clean_id, f"{clean_id}_1", f"{clean_id}_2", f"{clean_id}_3"]
        for v in variants:
            u = f"https://kamera.atlas.vegvesen.no/api/images/{v}"
            if u not in urls:
                urls.append(u)
    return urls


def get_nvdb_cameras():
    cameras = []
    seen_keys = set()
    tunnel_skipped = 0
    public_samples = []

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
                obj_id = str(obj.get("id", ""))
                egenskaper = obj.get("egenskaper", [])

                # Tunnel-Überwachungskameras aussortieren
                is_tunnel_cctv = any(
                    (eg.get("id") == 1179 or str(eg.get("navn", "")).lower() == "type kamera")
                    and str(eg.get("verdi", "")).lower() == "fast videokamera"
                    for eg in egenskaper
                )
                if is_tunnel_cctv:
                    tunnel_skipped += 1
                    continue

                if len(public_samples) < 3:
                    public_samples.append(egenskaper)

                direct_urls = []
                candidate_ids = []
                place = None

                for eg in egenskaper:
                    val = eg.get("verdi")
                    navn = str(eg.get("navn", "")).lower()
                    eg_type = str(eg.get("egenskapstype", ""))

                    if isinstance(val, str) and "http" in val:
                        direct_urls.append(val.strip())
                        if "/api/images/" in val:
                            candidate_ids.append(val.rstrip("/").split("/")[-1])
                        elif "id=" in val:
                            candidate_ids.append(val.split("id=")[-1].split("&")[0])
                    elif isinstance(val, str) and any(k in navn for k in ("stedsnavn", "navn", "beskrivelse", "plassering", "tittel")):
                        if not place and navn != "type kamera":
                            place = val.strip()
                    elif val is not None and eg_type in ("Tekst", "Heltall") and not any(bad in navn for bad in IGNORE_ID_KEYWORDS):
                        sval = str(val).strip()
                        # Typische Kamera-IDs sind Zahlen oder Muster wie 3001032_1
                        if re.fullmatch(r"[0-9]{4,12}(_[0-9]+)?", sval):
                            candidate_ids.append(sval)

                # Fallback auf NVDB-Objekt-ID, falls kein eigenes ID-Feld gefüllt war
                if not candidate_ids and not direct_urls and obj_id:
                    candidate_ids.append(obj_id)

                urls = build_image_urls(candidate_ids, direct_urls)
                if not urls:
                    continue

                primary_key = candidate_ids[0] if candidate_ids else urls[0]
                if primary_key in seen_keys:
                    continue
                seen_keys.add(primary_key)

                lok = obj.get("lokasjon", {})
                vegsys = lok.get("vegsystemreferanser", [])
                road = vegsys[0].get("kortform", "").split(" ")[0] if vegsys else ""
                fylke_ids = lok.get("fylker", [])
                region = FYLKER.get(fylke_ids[0], "Norwegen") if fylke_ids else "Norwegen"

                title = f"{place or 'Kamera ' + primary_key} – {road}" if road else (place or f"Kamera {primary_key}")
                cameras.append({
                    "id": primary_key,
                    "name": f"{title} ({region})",
                    "urls": urls
                })

            next_url = data.get("metadata", {}).get("neste", {}).get("href")
            url = next_url if next_url and next_url != url else None
        except Exception as e:
            print(f"NVDB-Hinweis: {e}")
            break

    print(f"NVDB-Filter: {tunnel_skipped} Tunnelkameras übersprungen, {len(cameras)} öffentliche Kamerastandorte gefunden.")
    if public_samples:
        print("DIAGNOSE (Öffentliche Webcam-Felder):", json.dumps(public_samples, ensure_ascii=False)[:900])

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
    print(f"Prüfe {len(cameras)} öffentliche Kameras in ganz Norwegen...")

    if not cameras:
        print("Abbruch: Keine Kameras gefunden.")
        return

    with ThreadPoolExecutor(max_workers=25) as executor:
        results_raw = list(executor.map(download_image, cameras))

    status_counts = Counter(status for status, _ in results_raw)
    print(f"Download-Statistik: {dict(status_counts)}")

    downloaded = [payload for status, payload in results_raw if status == "ok" and payload]
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
