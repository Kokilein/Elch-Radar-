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
    "Referer": "https://webkamera.atlas.vegvesen.no/"
}

model = YOLO("yolov8n.pt")
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def extract_cameras_from_json(obj, found=None, seen_urls=None):
    if found is None:
        found = []
    if seen_urls is None:
        seen_urls = set()

    if isinstance(obj, list):
        for item in obj:
            extract_cameras_from_json(item, found, seen_urls)
    elif isinstance(obj, dict):
        # Prüfen, ob dieses Objekt eine Kamera beschreibt
        cam_id = (
            obj.get("id")
            or obj.get("kameraId")
            or obj.get("cameraId")
            or obj.get("kameranummer")
        )
        img_url = None
        for k in ("url", "stillImageUrl", "imageUrl", "bildeUrl", "mediumUrl", "largeUrl", "href", "src"):
            val = obj.get(k)
            if isinstance(val, str) and (val.startswith("http") or val.startswith("/public/")):
                img_url = urljoin("https://webkamera.atlas.vegvesen.no/", val)
                break

        has_cam_hints = any(
            k in obj for k in (
                "stedsnavn", "beskrivelse", "vegnummer", "fylke",
                "breddegrad", "lengdegrad", "latitude", "longitude",
                "stillImageUrl", "bildeUrl", "kameraId", "cameraId"
            )
        )

        if not img_url and cam_id and has_cam_hints:
            img_url = f"https://webkamera.atlas.vegvesen.no/public/kamera?id={cam_id}"

        if img_url and ("kamera" in img_url or has_cam_hints) and img_url not in seen_urls:
            seen_urls.add(img_url)
            name = (
                obj.get("stedsnavn")
                or obj.get("navn")
                or obj.get("name")
                or obj.get("beskrivelse")
                or obj.get("description")
                or obj.get("tittel")
                or f"Kamera {cam_id}"
            )
            road = obj.get("vegnummer") or obj.get("veg") or ""
            fylke = obj.get("fylke") or obj.get("fylkesnavn") or obj.get("region") or "Norwegen"
            title = f"{name} – {road}" if road else str(name)
            found.append({
                "id": str(cam_id or len(found)),
                "name": f"{title} ({fylke})",
                "url": img_url
            })

        for val in obj.values():
            if isinstance(val, (dict, list)):
                extract_cameras_from_json(val, found, seen_urls)

    return found


def get_all_cameras():
    base_url = "https://webkamera.atlas.vegvesen.no/"
    candidate_urls = []
    js_snippets = []

    # 1. Startseite der Vegkamera-App laden und eingebundene JS-Datei auslesen
    try:
        home = requests.get(base_url, headers=HEADERS, timeout=20)
        js_paths = re.findall(r'src=["\']([^"\']+\.js)["\']', home.text)
        for js_path in js_paths:
            js_url = urljoin(base_url, js_path)
            js_res = requests.get(js_url, headers=HEADERS, timeout=20)
            if js_res.status_code == 200:
                js_text = js_res.text
                # Alle relevanten API-Pfade und Vegvesen-URLs aus dem Frontend-Code fischen
                abs_urls = re.findall(r'https?://[a-zA-Z0-9.-]*vegvesen\.no[^"\'\s`\\)]*', js_text)
                rel_paths = re.findall(r'["\'`](/(?:public|api|trafikk|ws|v1|v2|graphql)[^"\'\s`\\)]*)', js_text)

                for u in abs_urls:
                    if not any(u.endswith(ext) for ext in (".css", ".js", ".png", ".svg", ".ico", ".woff", ".woff2")):
                        if u not in candidate_urls:
                            candidate_urls.append(u)

                for p in rel_paths:
                    full_u = urljoin(base_url, p)
                    if full_u not in candidate_urls:
                        candidate_urls.append(full_u)

                # Kontext-Snippets sichern, falls die API spezielle Parameter braucht
                for m in re.finditer(r'(?:public/|vegvesen\.no/)', js_text):
                    start = max(0, m.start() - 80)
                    end = min(len(js_text), m.end() + 140)
                    snip = js_text[start:end]
                    if snip not in js_snippets and len(js_snippets) < 12:
                        js_snippets.append(snip)
    except Exception as e:
        print(f"Fehler beim Auslesen der Vegkamera-App: {e}")

    print(f"Gefundene API-Kandidaten in der App: {candidate_urls}")

    # 2. Gefundene API-Endpunkte abfragen
    for url in candidate_urls:
        if "kamera?id=" in url or url.endswith("/kamera"):
            continue
        try:
            res = requests.get(url, headers=HEADERS, timeout=20)
            if res.status_code == 200 and res.text.strip().startswith(("{", "[")):
                cams = extract_cameras_from_json(res.json())
                if cams:
                    print(f"Erfolgreich {len(cams)} aktive Verkehrskameras geladen über: {url}")
                    return cams
        except Exception as e:
            print(f"Info bei {url}: {e}")

    if js_snippets:
        print("DEBUG JS-Snippets:", " || ".join(js_snippets))

    return []


def download_image(cam):
    try:
        r = requests.get(cam["url"], headers=HEADERS, timeout=12)
        if r.status_code == 200 and len(r.content) > 5000 and not r.content[:20].lstrip().startswith(b"<"):
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
