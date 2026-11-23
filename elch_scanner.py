import os
import re
import json
import xml.etree.ElementTree as ET
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "X-Client": "elch-radar-norwegen"
}

model = YOLO("yolov8n.pt")
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def parse_vegvesen_metadata(raw_text):
    cameras = []
    seen_urls = set()
    clean_text = raw_text.lstrip("\ufeff \t\r\n")

    # 1. XML-Parsing (repariert vorher unmaskierte '&'-Zeichen in Ortsnamen/URLs)
    fixed_xml = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)", "&amp;", clean_text)
    try:
        root = ET.fromstring(fixed_xml)
        for elem in root.iter():
            fields = {k.lower(): v.strip() for k, v in elem.attrib.items() if v and v.strip()}
            for child in elem:
                ctag = child.tag.split("}")[-1].lower()
                if child.text and child.text.strip():
                    fields[ctag] = child.text.strip()

            cam_id = (
                fields.get("id")
                or fields.get("kameraid")
                or fields.get("cameraid")
                or fields.get("kameranummer")
            )
            img_url = None
            for val in fields.values():
                if "http://" in val or "https://" in val:
                    img_url = val
                    break

            tag = elem.tag.split("}")[-1].lower()
            is_cam_tag = tag in ("kamera", "webkamera", "camera", "webcam", "kameraobjekt")
            if (is_cam_tag or img_url) and (cam_id or img_url):
                if not img_url and cam_id:
                    img_url = f"https://webkamera.atlas.vegvesen.no/public/kamera?id={cam_id}"

                name = (
                    fields.get("stedsnavn")
                    or fields.get("navn")
                    or fields.get("name")
                    or fields.get("beskrivelse")
                    or fields.get("tittel")
                    or f"Kamera {cam_id}"
                )
                road = fields.get("vegnummer") or fields.get("veg") or ""
                fylke = fields.get("fylke") or fields.get("fylkesnavn") or fields.get("region") or "Norwegen"
                title = f"{name} – {road}" if road else name

                if img_url and img_url not in seen_urls:
                    seen_urls.add(img_url)
                    cameras.append({
                        "id": str(cam_id or len(cameras)),
                        "name": f"{title} ({fylke})",
                        "url": img_url
                    })
    except Exception as e:
        print(f"XML-Parser-Hinweis: {e}")

    # 2. Fallback: Kamera-URLs direkt per Suchmuster aus dem Text ziehen
    if not cameras:
        for match in re.finditer(r"https?://webkamera\.atlas\.vegvesen\.no/public/kamera\?id=(\d+)", clean_text):
            cam_id = match.group(1)
            img_url = match.group(0)
            if img_url not in seen_urls:
                seen_urls.add(img_url)
                cameras.append({
                    "id": cam_id,
                    "name": f"Kamera {cam_id} (Norwegen)",
                    "url": img_url
                })

    return cameras


def get_all_cameras():
    meta_url = "https://webkamera.atlas.vegvesen.no/public/metadata"
    try:
        res = requests.get(meta_url, headers=HEADERS, timeout=25)
        if res.status_code == 200 and res.text.strip():
            cams = parse_vegvesen_metadata(res.text)
            if cams:
                print(f"Erfolgreich {len(cams)} aktive Verkehrskameras geladen!")
                return cams
            else:
                print("DEBUG metadata-Inhalt:", res.text.strip()[:700])
    except Exception as e:
        print(f"Fehler beim Abruf: {e}")

    return []


def download_image(cam):
    try:
        r = requests.get(cam["url"], headers={"User-Agent": HEADERS["User-Agent"]}, timeout=12)
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

    if not downloaded and cameras:
        test_r = requests.get(cameras[0]["url"], headers={"User-Agent": HEADERS["User-Agent"]}, timeout=10)
        print(f"DEBUG Beispiel-Kamera ({cameras[0]['url']}): Status {test_r.status_code}, Größe {len(test_r.content)} Bytes")

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
