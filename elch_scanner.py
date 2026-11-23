import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# Offizielle API für alle Verkehrskameras in Norwegen
API_URL = "https://vegvesen.no/trafikkbeta/api/camera/all"
# Fallback-Quelle falls die Beta-API angepasst wird:
GEOJSON_URL = "https://www.vegvesen.no/ws/no/vegvesen/veg/trafikk/webkamera/v1/kamera"

model = YOLO("yolov8n.pt")

# YOLO Klassen für große Tiere (17=Pferd, 18=Schaf, 19=Kuh, 21=Bär -> darauf schlägt YOLO bei Elchen/Rentieren an)
ANIMAL_CLASSES = [17, 18, 19, 21]

# Gedächtnis gegen Spam (speichert bereits gemeldete Bild-URLs)
HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()

def get_all_cameras():
    cameras = []
    try:
        # Offizielle Vegvesen Kamera-Liste abrufen
        res = requests.get("https://webkamera.atlas.vegvesen.no/public/kameraer", timeout=20)
        if res.status_code == 200:
            data = res.json()
            for cam in data:
                cam_id = cam.get("id")
                name = cam.get("stedsnavn") or cam.get("beskrivelse") or f"Kamera {cam_id}"
                fylke = cam.get("fylke", "Norwegen")
                if cam_id:
                    cameras.append({
                        "id": str(cam_id),
                        "name": f"{name} ({fylke})",
                        "url": f"https://webkamera.atlas.vegvesen.no/public/kamera?id={cam_id}"
                    })
    except Exception as e:
        print(f"Fehler beim Laden der Kameraliste: {e}")
    return cameras

def download_image(cam):
    try:
        r = requests.get(cam["url"], timeout=10)
        if r.status_code == 200 and len(r.content) > 5000:
            # ETag oder Last-Modified nutzen, um doppelte Bilder zu vermeiden
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

    # 20 Kameras gleichzeitig herunterladen für maximale Geschwindigkeit
    with ThreadPoolExecutor(max_workers=20) as executor:
        downloaded = list(filter(None, executor.map(download_image, cameras)))

    print(f"{len(downloaded)} neue Kamerabilder geladen. Starte Elch-Erkennung...")

    for cam, filename, img_hash in downloaded:
        try:
            # Hohe Konfidenz (0.45), um Fehlalarme durch Schneehaufen/Steine zu minimieren
            results = model(filename, conf=0.45, verbose=False)
            for r in results:
                detected = [int(box.cls[0]) for box in r.boxes]
                if any(cls in ANIMAL_CLASSES for cls in detected):
                    out_file = f"hit_{cam['id']}.jpg"
                    r.save(filename=out_file)
                    caption = (
                        f"🦌 **Mögliche Elch-Sichtung!**\n"
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

    # Nur die letzten 2000 Hashes merken, damit die Datei klein bleibt
    with open(HISTORY_FILE, "w") as f:
        json.dump(list(seen_images)[-2000:], f)

if __name__ == "__main__":
    main()
