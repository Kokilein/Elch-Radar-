import os
import re
import json
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
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

# Offizielle Datex-II-Schnittstellen von Statens vegvesen für alle Verkehrskameras
DATEX_CAMERA_URLS = [
    "https://datex-server-get-v3-1.atlas.vegvesen.no/datexapi/GetCCTVSiteTable",
    "https://datex-server-get-v3-1.atlas.vegvesen.no/datexapi/GetCCTVSiteTable/pullsnapshotdata",
    "https://www.vegvesen.no/ws/no/vegvesen/veg/trafikkpublikasjon/kamera/2/GetCCTVSiteTable",
    "https://www.vegvesen.no/ws/no/vegvesen/veg/trafikkpublikasjon/kamera/3/GetCCTVSiteTable",
]

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
        primary = f"https://kamera.atlas.vegvesen.no/api/images/{clean_id}"
        if primary not in urls:
            urls.append(primary)
        if "_" not in clean_id:
            alt = f"https://kamera.atlas.vegvesen.no/api/images/{clean_id}_1"
            if alt not in urls:
                urls.append(alt)
    return urls


def parse_datex_xml(xml_text):
    cameras = []
    seen_ids = set()
    clean_xml = xml_text.lstrip("\ufeff \t\r\n")
    root = ET.fromstring(clean_xml)

    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag not in ("cctvcamerametadatarecord", "cctvcamerastatus"):
            continue

        cam_id = None
        name = None
        direct_url = None
        road = None

        for child in elem.iter():
            ctag = child.tag.split("}")[-1].lower()
            val = child.text.strip() if child.text else ""
            if not val:
                continue

            if ctag in ("cctvcameraidentification", "id") and not cam_id:
                cam_id = val
            elif ctag in ("value", "cctvcamerasitelocaldescription", "cctvcameraname") and not name:
                if not val.startswith("http"):
                    name = val
            elif ctag in ("urllinkaddress", "url") and "http" in val:
                direct_url = val
            elif ctag in ("roadnumber", "vegnummer") and not road:
                road = val

        if not cam_id and direct_url and "/api/images/" in direct_url:
            cam_id = direct_url.rstrip("/").split("/")[-1]

        if cam_id and cam_id not in seen_ids:
            seen_ids.add(cam_id)
            title = f"{name or 'Kamera ' + cam_id} – {road}" if road else (name or f"Kamera {cam_id}")
            cameras.append({
                "id": cam_id,
                "name": f"{title} (Norwegen)",
                "urls": build_image_urls(cam_id, direct_url)
            })

    # Fallback: direkte Bild-Links aus dem XML fischen
    if not cameras:
        for match in re.finditer(r"https://kamera\.atlas\.vegvesen\.no/api/images/([a-zA-Z0-9_-]+)", clean_xml):
            cam_id = match.group(1)
            if cam_id not in seen_ids:
                seen_ids.add(cam_id)
                cameras.append({
                    "id": cam_id,
                    "name": f"Kamera {cam_id} (Norwegen)",
                    "urls": [match.group(0)]
                })

    return cameras


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
                break
            data = res.json()
            for obj in data.get("objekter", []):
                obj_id = str(obj.get("id", ""))
                ext_id = None
                direct_url = None
                place = None

                for eg in obj.get("egenskaper", []):
                    val = eg.get("verdi")
                    navn = str(eg.get("navn", "")).lower()
                    if isinstance(val, str) and "http" in val:
                        direct_url = val
                        if "/api/images/" in val or "id=" in val:
                            ext_id = val.split("id=")[-1].split("/")[-1]
                    elif any(k in navn for k in ("kamera", "ekstern", "nummer", "id")) and val:
                        ext_id = str(val)
                    elif isinstance(val, str) and any(k in navn for k in ("stedsnavn", "navn", "beskrivelse", "plassering")):
                        place = place or val

                cam_id = ext_id or obj_id
                if not cam_id or cam_id in seen_ids:
                    continue
                seen_ids.add(cam_id)

                lok = obj.get("lokasjon", {})
                vegsys = lok.get("vegsystemreferanser", [])
                road = vegsys[0].get("kortform", "").split(" ")[0] if vegsys else ""
                fylke_ids = lok.get("fylker", [])
                region = FYLKER.get(fylke_ids[0], "Norwegen") if fylke_ids else "Norwegen"

                title = f"{place or 'Kamera ' + cam_id} – {road}" if road else (place or f"Kamera {cam_id}")
                cameras.append({
                    "id": cam_id,
                    "name": f"{title} ({region})",
                    "urls": build_image_urls(cam_id, direct_url)
                })

            next_url = data.get("metadata", {}).get("neste", {}).get("href")
            url = next_url if next_url and next_url != url else None
        except Exception as e:
            print(f"NVDB-Hinweis: {e}")
            break

    return cameras


def get_all_cameras():
    # 1. Datex-II Gesamtliste aller Kameras prüfen
    for endpoint in DATEX_CAMERA_URLS:
        try:
            res = requests.get(endpoint, headers=HEADERS, timeout=25)
            if res.status_code == 200 and res.text.strip().startswith("<"):
                cams = parse_datex_xml(res.text)
                if cams:
                    print(f"Erfolgreich {len(cams)} Kameras geladen über Datex: {endpoint}")
                    return cams
            else:
                print(f"Info Datex {endpoint}: Status {res.status_code}")
        except Exception as e:
            print(f"Info Datex {endpoint}: {e}")

    # 2. Fallback: NVDB-Kameras mit neuem Bild-Server
    nvdb_cams = get_nvdb_cameras()
    if nvdb_cams:
        print(f"Erfolgreich {len(nvdb_cams)} Kameras aus NVDB geladen!")
        return nvdb_cams

    return []


def download_image(cam):
    for url in cam["urls"]:
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code == 200 and len(r.content) > 5000 and not r.content[:20].lstrip().startswith(b"<"):
                img_hash = f"{cam['id']}_{r.headers.get('Last-Modified', len(r.content))}"
                if img_hash in seen_images:
                    return None
                safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", cam["id"])
                filename = f"cam_{safe_id}.jpg"
                with open(filename, "wb") as f:
                    f.write(r.content)
                cam["active_url"] = url
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
