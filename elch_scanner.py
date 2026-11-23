import os
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


def parse_vegvesen_xml(xml_text):
    cameras = []
    seen_urls = set()
    clean_xml = xml_text.lstrip("\ufeff \t\r\n")
    root = ET.fromstring(clean_xml)

    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag not in ("kamera", "webkamera", "camera", "webcam", "item"):
            continue

        fields = {k.lower(): v for k, v in elem.attrib.items() if v}
        for child in elem.iter():
            ctag = child.tag.split("}")[-1].lower()
            if child.text and child.text.strip():
                fields.setdefault(ctag, child.text.strip())

        cam_id = (
            fields.get("id")
            or fields.get("kameraid")
            or fields.get("cameraid")
            or fields.get("kameranummer")
        )

        img_url = None
        for val in fields.values():
            if isinstance(val, str) and ("http://" in val or "https://" in val):
                img_url = val
                break

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
        full_name = f"{title} ({fylke})"

        if img_url and img_url not in seen_urls:
            seen_urls.add(img_url)
            cameras.append({
                "id": str(cam_id or len(cameras)),
                "name": full_name,
                "url": img_url
            })

    return cameras


def get_all_cameras():
    # 1. Offizieller XML-Metadata-Feed von Statens vegvesen
    meta_url = "https://webkamera.atlas.vegvesen.no/public/metadata"
    try:
        res = requests.get(meta_url, headers=HEADERS, timeout=25)
        if res.status_code == 200 and res.text.strip():
            text = res.text.lstrip("\ufeff \t\r\n")
            if text.startswith("<"):
                cams = parse_vegvesen_xml(text)
                if cams:
                    print(f"Erfolgreich {len(cams)} Kameras aus Vegvesen-XML geladen!")
                    return cams
                else:
                    print("DEBUG XML-Anfang:", text[:600])
            else:
                print("DEBUG Antwort-Anfang:", text[:600])
    except Exception as e:
        print(f"Fehler beim XML-Abruf: {e}")

    # 2. Fallback: NVDB API v4 Objekttyp 163 (Kamera)
    cameras = []
    seen_urls = set()
    url = "https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/163?inkluder=egenskaper,lokasjon&antall=1000"
    nvdb_headers = {**HEADERS, "Accept": "application/vnd.vegvesen.nvdb-v4+json, application/json"}

    while url:
        try:
            res = requests.get(url, headers=nvdb_headers, timeout=25)
            if res.status_code != 200:
                break
            data = res.json()
            for obj in data.get("objekter", []):
                cam_id = str(obj.get("id", ""))
                img_url = None
                web_cam_id = None
                place = None

                for eg in obj.get("egenskaper", []):
                    val = eg.get("verdi")
                    navn = str(eg.get("navn", "")).lower()
                    if isinstance(val, str) and ("http" in val or "vegvesen.no" in val):
                        img_url = val if val.startswith("http") else f"https://{val.lstrip('/')}"
                    elif any(k in navn for k in ("kamera", "ekstern", "nummer")) and val:
                        web_cam_id = str(val)
                    elif isinstance(val, str) and any(k in navn for k in ("stedsnavn", "navn", "beskrivelse", "plassering")):
                        place = place or val

                if not img_url and web_cam_id:
                    img_url = f"https://webkamera.atlas.vegvesen.no/public/kamera?id={web_cam_id}"

                lok = obj.get("lokasjon", {})
                fylke_ids = lok.get("fylker", [])
                region = FYLKER.get(fylke_ids[0], "Norwegen") if fylke_ids else "Norwegen"

                if img_url and img_url not in seen_urls:
                    seen_urls.add(img_url)
                    cameras.append({
                        "id": cam_id,
                        "name": f"{place or 'Kamera ' + cam_id} ({region})",
                        "url": img_url
                    })

            next_url = data.get("metadata", {}).get("neste", {}).get("href")
            url = next_url if next_url and next_url != url else None
        except Exception:
            break

    return cameras


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
