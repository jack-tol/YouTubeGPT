import json
import re
from datetime import datetime, timezone
from html import unescape
from xml.etree import ElementTree

import requests


def format_published_date(date_str):
    if not date_str:
        return ""
    dt = datetime.fromisoformat(date_str).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + "Z"


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def scrape_video_data(video_id):
    session = requests.Session()

    page_html = unescape(
        session.get(f"https://www.youtube.com/watch?v={video_id}").text
    )

    ld_match = re.search(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', page_html, re.DOTALL)
    ld_data = json.loads(ld_match.group(1)) if ld_match else {}

    description = ""
    key = '"shortDescription":"'
    idx = page_html.find(key)
    if idx >= 0:
        start = idx + len(key)
        i = start
        while i < len(page_html) and i < start + 50000:
            if page_html[i] == '"' and page_html[i - 1] != '\\':
                break
            i += 1
        description = json.loads('"' + page_html[start:i] + '"')
        description = re.sub(r'[\u2060\u200b\u200c\u200d\ufeff\u00a0]', ' ', description)
        description = re.sub(r'[\r\n]+', ' ', description)
        description = re.sub(r' {2,}', ' ', description).strip()

    channel_match = re.search(r'<link itemprop="name" content="([^"]+)"', page_html)

    length_match = re.search(r'"lengthSeconds":"(\d+)"', page_html)
    duration = format_time(int(length_match.group(1))) if length_match else ""

    tags = re.findall(r'<meta property="og:video:tag" content="([^"]*?)"', page_html)

    api_key = re.search(r'"INNERTUBE_API_KEY":\s*"([^"]+)"', page_html).group(1)

    player_response = session.post(
        f"https://www.youtube.com/youtubei/v1/player?key={api_key}",
        json={
            "context": {"client": {"clientName": "ANDROID", "clientVersion": "20.10.38"}},
            "videoId": video_id,
        },
    ).json()

    caption_tracks = (
        player_response
        .get("captions", {})
        .get("playerCaptionsTracklistRenderer", {})
        .get("captionTracks", [])
    )

    transcript = ""
    if caption_tracks:
        english_tracks = [t for t in caption_tracks if t.get("languageCode") == "en"]
        manual = [t for t in english_tracks if t.get("kind") != "asr"]
        track = manual[0] if manual else english_tracks[0]

        xml_data = session.get(track["baseUrl"]).text
        root = ElementTree.fromstring(xml_data)

        parts = []
        for p in root.iter("p"):
            start_ms = int(p.attrib.get("t", "0"))
            words = []
            for s in p.iter("s"):
                if s.text:
                    words.append(unescape(s.text))
            if words:
                parts.append(f"[{format_time(start_ms / 1000.0)}] {''.join(words).strip()}")

        transcript = " ".join(parts)

    return {
        "video_id": video_id,
        "video_title": ld_data.get("name", ""),
        "channel_name": channel_match.group(1) if channel_match else "",
        "video_description": description,
        "video_published_date": format_published_date(ld_data.get("uploadDate", "")),
        "video_duration": duration,
        "video_tags": tags,
        "video_transcript": transcript,
    }