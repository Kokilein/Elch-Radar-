import os
import re
import json
from urllib.parse import urljoin
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.vegvesen.no/"
}

# Die neuen Endpunkte von Statens vegvesen (kamera.atlas.vegvesen.no)
CAMERA_LIST_ENDPOINTS = [
    "https://kamera.atlas.vegvesen.no/api/cameras",
    "https://kamera.atlas.vegvesen.no/api/v1/cameras",
    "https://kamera.atlas.vegvesen.no/api/v2/cameras",
    "https://kamera.atlas.vegvesen.no/api/kameraer",
    "https://kamera.atlas.vegvesen.no/api/images",
    "https://www.vegvesen.no/trafikk/api/cameras",
    "https://www.vegvesen.no/trafikk/api/kameraer",
]

model = YOLO("yolov8n.pt")
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def parse_camera_list(data):
    cameras = []
    seen_ids = set()

    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("cameras", "kameraer", "items", "data", "features", "results"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        if not items and data and all(isinstance(v, dict) for v in data.values()):
            for k, v in data.items():
                entry = dict(v)
                entry.setdefault("id", k)
                items.append(entry)

    for item in items:
        if not isinstance(item, dict):
            continue
        props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        merged = {**item, **props}

        cam_id = (
            merged.get("id")
            or merged.get("cameraId")
            or merged.get("kameraId")
            or merged.get("kameranummer")
        )
        if not cam_id:
            continue

        cam_id_str = str(cam_id).strip()
        if cam_id_str in seen_ids:
            continue
        seen_ids.add(cam_id_str)

        # Standard-Bild-URL auf dem neuen Server
        img_url = f"https://kamera.atlas.vegvesen.no/api/images/{cam_id_str}"
        for url_key in ("imageUrl", "stillImageUrl", "url", "bildeUrl", "href"):
            val = merged.get(url_key)
            if isinstance(val, str) and (val.startswith("http") or val.startswith("/api/")):
                img_url = urljoin("https://kamera.atlas.vegvesen.no/", val)
                break

        name = (
            merged.get("name")
            or merged.get("stedsnavn")
            or merged.get("navn")
            or merged.get("description")
            or merged.get("beskrivelse")
            or merged.get("title")
            or f"Kamera {cam_id_str}"
        )
        road = merged.get("road") or merged.get("roadNumber") or merged.get("vegnummer") or merged.get("veg") or ""
        fylke = merged.get("county") or merged.get("fylke") or merged.get("region") or "Norwegen"

        title = f"{name} – {road}" if road else str(name)
        cameras.append({
            "id": cam_id_str,
            "name": f"{title} ({fylke})",
            "url": img_url
        })

    return cameras


def get_all_cameras():
    # 1. Direktabruf der Kameraliste vom neuen Server
    for endpoint in CAMERA_LIST_ENDPOINTS:
        try:
            res = requests.get(endpoint, headers=HEADERS, timeout=20)
            if res.status_code == 200 and res.text.strip().startswith(("{", "[")):
                cams = parse_camera_list(res.json())
                if cams:
                    print(f"Erfolgreich {len(cams)} aktive Verkehrskameras geladen über: {endpoint}")
                    return cams
            else:
                print(f"Info {endpoint}: Status {res.status_code}")
        except Exception as e:
            print(f"Info {endpoint}: {e}")

    # 2. Fallback: Die Vegvesen-Verkehrskarte nach dem genauen API-Pfad durchsuchen
    try:
        map_url = "https://www.vegvesen.no/trafikk/hvaskjer?layers=ctv"
        page = requests.get(map_url, headers=HEADERS, timeout=20)
        js_paths = re.findall(r'src=["\']([^"\']+\.js)["\']', page.text)
        found_apis = []
        for js_path in js_paths:
            js_res = requests.get(urljoin(map_url, js_path), headers=HEADERS, timeout=20)
            if js_res.status_code == 200:
                matches = re.findall(r'https?://[a-zA-Z0-9.-]*vegvesen\.no/[^"\'\s`\\)]*(?:camera|kamera|ctv)[^"\'\s`\\)]*', js_res.text, re.I)
                for m in matches:
                    if m not in found_apis:
                        found_apis.append(m)
        if found_apis:
            print("DEBUG Gefundene Kamera-APIs auf der Verkehrskarte:", found_apis)
            for api in found_apis:
                try:
                    r = requests.get(api, headers=HEADERS, timeout=15)
                    if r.status_code == 200 and r.text.strip().startswith(("{", "[")):
                        cams = parse_camera_list(r.json())
                        if cams:
                            print(f"Erfolgreich {len(cams)} Kameras geladen über: {api}")
                            return cams
                except Exception:
                    pass
    except Exception as e:
        print(f"Fallback-Hinweis: {e}")

    return []


def download_image(cam):
    try:
        r = requests.get(cam["url"], headers=HEADERS, timeout=12)
        if r.status_code == 200 and len(r.content) > 5000 and not r.content[:20].lstrip().startswith(b"<"):
            img_hash = f"{cam['id']}_{r.headers.get('Last-Modified', len(r.content))}"
            if img_hash in seen_images:
                return None
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", cam["id"])
            filename = f"cam_{safe_id}.jpg"
            with open(filename, "wb") as f:
                f.write(r.content)
            return (cam, filename, img_hash)
    except Exception:
        pass
    return None


def send_telegram(image_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": photo})


def main():
    cameras = get_all_cameras()
    print(f"Prüfe {len(cameras)} Kameras in ganz Norwegen...")

    if not cameras:
        print("Abbruch: Keine Kameras gefunden.")
        return

    with ThreadPoolExecutor(max_workers=25) as executor:
        downloaded = list(filter(None, executor.map(download_image, cameras)))

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
                        f"🔗 Live-Bild: {cam['url']}"
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
