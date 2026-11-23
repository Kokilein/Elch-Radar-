import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "X-Client": "elch-radar-norwegen"
}

# Offizielle Vegvesen- und NVDB-Endpunkte für alle Verkehrskameras in Norwegen
CAMERA_ENDPOINTS = [
    "https://www.vegvesen.no/trafikk/api/camera/all",
    "https://www.vegvesen.no/trafikkbeta/api/camera/all",
    "https://www.vegvesen.no/ws/no/vegvesen/veg/trafikk/webkamera/v1/kamera",
    "https://webkamera.atlas.vegvesen.no/public/kameraer",
    "https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/67?inkluder=egenskaper,vegsystemreferanse&antall=2000",
    "https://nvdbapiles-v3.atlas.vegvesen.no/vegobjekter/api/v3/vegobjekter/67?inkluder=egenskaper,vegsystemreferanse&antall=2000",
]

model = YOLO("yolov8n.pt")

# YOLO Klassen für große Tiere (17=Pferd, 18=Schaf, 19=Kuh, 21=Bär -> darauf schlägt YOLO bei Elchen/Rentieren an)
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def parse_cameras(data):
    cameras = []
    seen_urls = set()

    # Fall 1: NVDB-API Format (Nasjonal vegdatabank, Objekttyp 67 = Kamera)
    if isinstance(data, dict) and "objekter" in data:
        for obj in data["objekter"]:
            cam_id = str(obj.get("id", ""))
            img_url = None
            place = None

            for eg in obj.get("egenskaper", []):
                val = eg.get("verdi")
                navn = str(eg.get("navn", ""))
                if isinstance(val, str) and val.startswith("http"):
                    img_url = val
                elif isinstance(val, str) and navn in ("Stedsnavn", "Navn", "Beskrivelse", "Tittel"):
                    place = val

            road = obj.get("vegsystemreferanse", {}).get("kortform", "")
            label = f"{place or 'Verkehrskamera'} ({road})".strip() if road else (place or f"Kamera {cam_id}")

            if img_url and img_url not in seen_urls:
                seen_urls.add(img_url)
                cameras.append({"id": cam_id, "name": label, "url": img_url})
        return cameras

    # Fall 2: Standard-Liste, GeoJSON oder verschachteltes JSON
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("features", "cameras", "kameraer", "items", "data"):
            if isinstance(data.get(key), list):
                items = data[key]
                break

    for item in items:
        if not isinstance(item, dict):
            continue
        props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        merged = {**item, **props}

        cam_id = merged.get("id") or merged.get("kameraId") or merged.get("cameraId")
        img_url = None
        for url_key in ("url", "stillImageUrl", "imageUrl", "bildeUrl", "href"):
            val = merged.get(url_key)
            if isinstance(val, str) and val.startswith("http"):
                img_url = val
                break

        if not img_url and cam_id:
            img_url = f"https://webkamera.atlas.vegvesen.no/public/kamera?id={cam_id}"

        name = (
            merged.get("stedsnavn")
            or merged.get("name")
            or merged.get("navn")
            or merged.get("beskrivelse")
            or merged.get("description")
            or merged.get("title")
            or f"Kamera {cam_id}"
        )
        fylke = merged.get("fylke") or merged.get("county") or merged.get("region") or "Norwegen"

        if img_url and img_url not in seen_urls:
            seen_urls.add(img_url)
            cameras.append({
                "id": str(cam_id or len(cameras)),
                "name": f"{name} ({fylke})",
                "url": img_url
            })

    return cameras


def get_all_cameras():
    for endpoint in CAMERA_ENDPOINTS:
        try:
            res = requests.get(endpoint, headers=HEADERS, timeout=20)
            if res.status_code != 200 or not res.text.strip():
                print(f"Überspringe {endpoint} (Status {res.status_code}, leer)")
                continue

            data = res.json()
            cameras = parse_cameras(data)
            if cameras:
                print(f"Erfolgreich {len(cameras)} Kameras geladen über: {endpoint}")
                return cameras
            else:
                print(f"Keine Kameras im Format von {endpoint} gefunden.")
        except Exception as e:
            print(f"Fehler bei {endpoint}: {e}")

    return []


def download_image(cam):
    try:
        r = requests.get(cam["url"], headers=HEADERS, timeout=10)
        content_type = r.headers.get("Content-Type", "")
        if r.status_code == 200 and len(r.content) > 5000 and "image" in content_type:
            img_hash = f"{cam['id']}_{r.headers.get('Last-Modified', len(r.content))}"
            if img_hash in seen_images:
                return None
            filename = f"cam_{cam['id']}.jpg"
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

    with ThreadPoolExecutor(max_workers=20) as executor:
        downloaded = list(filter(None, executor.map(download_image, cameras)))

    print(f"{len(downloaded)} neue Kamerabilder geladen. Starte Elch-Erkennung...")

    for cam, filename, img_hash in downloaded:
        try:
            results = model(filename, conf=0.45, verbose=False)
            for r in results:
                detected = [int(box.cls[0]) for box in r.boxes]
                if any(cls in ANIMAL_CLASSES for cls in detected):
                    out_file = f"hit_{cam['id']}.jpg"
                    r.save(filename=out_file)
                    caption = (
                        f"🦌 Mögliche Elch-Sichtung!\n"
                        f"📍 Ort: {cam['name']}\n"
                        f"🔗 Live-Bild: {cam['url']}"
                    )
                    send_telegram(out_file, caption)
                    seen_images.add(img_hash)
                    print(f"TREFFER gemeldet: {cam['name']}")
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
