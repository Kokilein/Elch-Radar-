import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "X-Client": "elch-radar-norwegen"
}

FYLKER = {
    3: "Oslo", 11: "Rogaland", 15: "Møre og Romsdal", 18: "Nordland",
    30: "Viken", 31: "Østfold", 32: "Akershus", 33: "Buskerud",
    34: "Innlandet", 38: "Vestfold og Telemark", 39: "Vestfold",
    40: "Telemark", 42: "Agder", 46: "Vestland", 50: "Trøndelag",
    54: "Troms og Finnmark", 55: "Troms", 56: "Finnmark"
}

DIRECT_WEBCAM_URLS = [
    "https://trafikkinfo.atlas.vegvesen.no/trafikkinfo/webkamera/v1/kamera",
    "https://trafikkinfo.atlas.vegvesen.no/trafikkinfo/webkamera/v2/kamera",
    "https://webkamera.atlas.vegvesen.no/public/metadata",
    "https://webkamera.atlas.vegvesen.no/public/kamera",
]

model = YOLO("yolov8n.pt")
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def parse_direct_webcams(data):
    cameras = []
    seen_urls = set()

    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("kameraer", "cameras", "features", "items", "data", "webkameraer"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        if not items and all(isinstance(v, dict) for v in data.values()):
            items = list(data.values())

    for item in items:
        if not isinstance(item, dict):
            continue
        props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        merged = {**item, **props}

        cam_id = merged.get("id") or merged.get("kameraId") or merged.get("cameraId") or merged.get("kameranummer")
        img_url = None
        for k, v in merged.items():
            if isinstance(v, str) and ("http" in v or "vegvesen.no" in v):
                img_url = v if v.startswith("http") else f"https://{v.lstrip('/')}"
                break

        if not img_url and cam_id:
            img_url = f"https://webkamera.atlas.vegvesen.no/public/kamera?id={cam_id}"

        name = (
            merged.get("stedsnavn")
            or merged.get("navn")
            or merged.get("name")
            or merged.get("beskrivelse")
            or merged.get("tittel")
            or f"Kamera {cam_id}"
        )
        fylke = merged.get("fylke") or merged.get("region") or "Norwegen"

        if img_url and img_url not in seen_urls:
            seen_urls.add(img_url)
            cameras.append({
                "id": str(cam_id or len(cameras)),
                "name": f"{name} ({fylke})",
                "url": img_url
            })

    return cameras


def get_all_cameras():
    # 1. Direkte Statens vegvesen Webkamera-APIs testen
    for endpoint in DIRECT_WEBCAM_URLS:
        try:
            res = requests.get(endpoint, headers=HEADERS, timeout=20)
            if res.status_code == 200 and res.text.strip().startswith(("{", "[")):
                cams = parse_direct_webcams(res.json())
                if cams:
                    print(f"Erfolgreich {len(cams)} Kameras geladen über: {endpoint}")
                    return cams
            else:
                print(f"Info {endpoint}: Status {res.status_code}")
        except Exception as e:
            print(f"Info {endpoint}: {e}")

    # 2. Fallback: NVDB API v4 (Objekttyp 67 = Kamera)
    cameras = []
    seen_urls = set()
    sample_debug = []
    url = "https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/67?inkluder=egenskaper,lokasjon&antall=1000"
    nvdb_headers = {**HEADERS, "Accept": "application/vnd.vegvesen.nvdb-v4+json, application/json"}

    while url:
        try:
            res = requests.get(url, headers=nvdb_headers, timeout=25)
            if res.status_code != 200:
                print(f"NVDB-Status: {res.status_code}")
                break

            data = res.json()
            objekter = data.get("objekter", [])
            if not objekter:
                break

            for obj in objekter:
                cam_id = str(obj.get("id", ""))
                egenskaper = obj.get("egenskaper", [])
                if len(sample_debug) < 2:
                    sample_debug.append(egenskaper)

                img_url = None
                web_cam_id = None
                place = None

                for eg in egenskaper:
                    val = eg.get("verdi")
                    navn = str(eg.get("navn", "")).lower()

                    if isinstance(val, str) and ("http" in val or "vegvesen.no" in val or "webkamera" in val):
                        img_url = val if val.startswith("http") else f"https://{val.lstrip('/')}"
                    elif any(k in navn for k in ("kamera", "ekstern", "nummer", "webkamera")) and val:
                        web_cam_id = str(val)
                    elif isinstance(val, str) and any(k in navn for k in ("stedsnavn", "navn", "beskrivelse", "plassering", "tittel")):
                        if not place:
                            place = val

                if not img_url and web_cam_id:
                    img_url = f"https://webkamera.atlas.vegvesen.no/public/kamera?id={web_cam_id}"

                lok = obj.get("lokasjon", {})
                vegsys = lok.get("vegsystemreferanser", [])
                road = vegsys[0].get("kortform", "").split(" ")[0] if vegsys else ""
                fylke_ids = lok.get("fylker", [])
                region = FYLKER.get(fylke_ids[0], "Norwegen") if fylke_ids else "Norwegen"

                title = place or f"Kamera {cam_id}"
                full_name = f"{title} – {road} ({region})" if road else f"{title} ({region})"

                if img_url and img_url not in seen_urls:
                    seen_urls.add(img_url)
                    cameras.append({
                        "id": cam_id,
                        "name": full_name,
                        "url": img_url
                    })

            next_url = data.get("metadata", {}).get("neste", {}).get("href")
            url = next_url if next_url and next_url != url else None

        except Exception as e:
            print(f"NVDB-Fehler: {e}")
            break

    if not cameras and sample_debug:
        print("DEBUG NVDB-Felder:", json.dumps(sample_debug, ensure_ascii=False)[:1000])

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
