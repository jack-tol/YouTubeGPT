import json
from tools import scrape_video_data

video_data = scrape_video_data("ORMx45xqWkA")

print(json.dumps(video_data, indent=4, ensure_ascii=False, sort_keys=True))