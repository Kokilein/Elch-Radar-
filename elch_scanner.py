import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/vnd.vegvesen.nvdb-v4+json, application/json",
    "X-Client": "elch-radar-norwegen"
}

# Norwegische Regionen (Fylker) für die Ortsangabe
FYLKER = {
    3: "Oslo", 11: "Rogaland", 15: "Møre og Romsdal", 18: "Nordland",
    30: "Viken", 31: "Østfold", 32: "Akershus", 33: "Buskerud",
    34: "Innlandet", 38: "Vestfold og Telemark", 39: "Vestfold",
    40: "Telemark", 42: "Agder", 46: "Vestland", 50: "Trøndelag",
    54: "Troms og Finnmark", 55: "Troms", 56: "Finnmark"
}

model = YOLO("yolov8n.pt")

# YOLO Klassen für große Tiere (17=Pferd, 18=Schaf, 19=Kuh, 21=Bär -> darauf schlägt YOLO bei Elchen/Rentieren an)
ANIMAL_CLASSES = [17, 18, 19, 21]

HISTORY_FILE = "seen_moose.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        seen_images = set(json.load(f))
else:
    seen_images = set()


def get_all_cameras():
    cameras = []
    seen_urls = set()
    # Objekttyp 67 in der Nasjonal vegdatabank (NVDB) = Verkehrskamera
    url = "https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/67?inkluder=egenskaper,lokasjon&antall=1000"

    while url:
        try:
            res = requests.get(url, headers=HEADERS, timeout=25)
            if res.status_code != 200:
                print(f"NVDB-Fehler (Status {res.status_code}): {res.text[:250]}")
                break

            data = res.json()
            objekter = data.get("objekter", [])
            if not objekter:
                break

            for obj in objekter:
                cam_id = str(obj.get("id", ""))
                img_url = None
                web_cam_id = None
                place = None

                for eg in obj.get("egenskaper", []):
                    val = eg.get("verdi")
                    navn = str(eg.get("navn", "")).lower()

                    if isinstance(val, str) and val.startswith("http"):
                        img_url = val
                    elif "kamera" in navn and "id" in navn and val:
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

            # Nächste Seite laden, falls es mehr als 1.000 Kameras gibt
            next_url = data.get("metadata", {}).get("neste", {}).get("href")
            url = next_url if next_url and next_url != url else None

        except Exception as e:
            print(f"Fehler beim Abruf der NVDB-Kameras: {e}")
            break

    return cameras


def download_image(cam):
    try:
        r = requests.get(cam["url"], headers={"User-Agent": HEADERS["User-Agent"]}, timeout=12)
        # Prüfen, ob die Datei groß genug ist und keine HTML-Fehlerseite zurückkam
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
