"""Fetch knowledge (text + image) from Wikipedia and cache it in data/wiki_cache.json."""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Ye line DIR ko define karti hai
DIR = Path(__file__).parent / "data" 
CACHE = DIR / "wiki_cache.json"
API = "https://en.wikipedia.org/w/api.php"
COMMONS = "https://commons.wikimedia.org/w/api.php"
STOP = {"See also", "References", "External links", "Further reading", "Notes", "Bibliography", "Sources", "Gallery"}

def _get(params: dict, api: str = API) -> dict:
    url = api + "?" + urllib.parse.urlencode({**params, "format": "json", "redirects": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "BanarasGhatBot/1.0 (learning project)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}

def _page(title: str):
    d = _get({"action": "query", "prop": "extracts|pageimages|info", "explaintext": 1, "exsectionformat": "plain",
              "pithumbsize": 800, "inprop": "url", "titles": title})
    p = next(iter(d.get("query", {}).get("pages", {}).values()), None)
    return None if not p or "missing" in p else p

def commons_image(query: str) -> str:
    d = _get({"action": "query", "prop": "generator", "gsrsearch": query, "gsrnamespace": 6, "gsrsearch": query,
              "gsrlimit": 8, "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 800}, COMMONS)
    pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") == "image/jpeg":
            return info.get("thumburl") or info.get("url", "")
    return ""

def _section(lines: list[str], keyword: str) -> list[str]:
    out = []
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            if len(line) < 50: 
                for nxt in lines[i + 1:]:
                    if len(nxt) < 50:
                        break
                    out.append(nxt)
            else:
                out.append(line)
    return list(dict.fromkeys(out))

def fetch(title: str, keyword: str | None = None, image_query: str | None = None) -> dict | None:
    p = _page(title)
    if p is None:
        return None
    lines = []
    for line in p.get("extract", "").splitlines():
        if line.strip() in STOP:
            break
        if line.strip():
            lines.append(line.strip())
    if keyword:
        lines = _section(lines, keyword)
    image = "" if keyword else p.get("thumbnail", {}).get("source", "")
    
    if not image and image_query:
        image = commons_image(image_query)
    return {"title": p["title"], "url": p.get("fullurl", ""), "text": "\n".join(lines)[:5000], "image": image}

def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

def build_cache(places: list[dict]) -> dict:
    cache = load_cache()
    for p in places:
        if p["id"] in cache or not (p.get("wiki") or p.get("image_query")):
            continue
        print(f"Fetching: {p['wiki'] or p['name']}...")
        if p.get("wiki"):
            res = fetch(p["wiki"], p.get("keyword"), p.get("image_query"))
        else:
            res = {"title": p["name"], "url": "", "text": "", "image": commons_image(p["image_query"])}
        
        if not res: 
            res = {"title": p["name"], "url": "", "text": "",
                   "image": commons_image(p.get("image_query") or p["name"] + " Varanasi")}
        
        cache[p["id"]] = res
        try:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError: 
            pass
        time.sleep(1)
    return cache

if __name__ == "__main__":
    print("🚀 Script start ho rahi hai...")
    try:
        places_file = DIR / "places.json"
        print(f"📂 Places file check kar rahe hain: {places_file}")
        
        places_data = json.loads(places_file.read_text(encoding="utf-8"))
        print(f"✅ Total {len(places_data)} places mile.")
        
        print("⏳ Wikipedia se data fetch karna shuru kar rahe hain (Isko 2-3 minute lag sakte hain)...")
        build_cache(places_data)
        
        print("🎉 Data fetch complete! Ab aap Streamlit chala sakte hain.")
        for k, v in load_cache().items():
            print(f" - {k} -> {v['title']}")
            
    except Exception as e:
        print(f"❌ Error aagaya bhai: {e}")