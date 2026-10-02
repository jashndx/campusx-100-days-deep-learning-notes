import json
import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

def fetch_all_metadata():
    with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\playlist_summary.json', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = data['entries']
    print(f"Loaded {len(entries)} entries.")
    
    metadata_cache_path = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\videos_metadata_raw.json'
    existing = {}
    if os.path.exists(metadata_cache_path):
        with open(metadata_cache_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"Already cached: {len(existing)} videos.")
        
    ydl_opts = {'quiet': True, 'skip_download': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for idx, entry in enumerate(entries):
            vid = entry['id']
            if vid in existing and existing[vid].get('description'):
                continue
            title = entry.get('title', '')
            print(f"[{idx+1}/{len(entries)}] Fetching metadata for {vid}: {title[:40]}...")
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                existing[vid] = {
                    'id': vid,
                    'index': idx + 1,
                    'title': info.get('title', title),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', ''),
                    'webpage_url': info.get('webpage_url', f"https://www.youtube.com/watch?v={vid}"),
                    'tags': info.get('tags', [])
                }
            except Exception as e:
                print(f"  Error fetching metadata for {vid}: {e}")
                existing[vid] = {
                    'id': vid,
                    'index': idx + 1,
                    'title': title,
                    'description': '',
                    'duration': 0,
                    'upload_date': '',
                    'webpage_url': f"https://www.youtube.com/watch?v={vid}",
                    'tags': []
                }
            if (idx + 1) % 10 == 0:
                with open(metadata_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)
                print(f"  Saved progress ({len(existing)}/{len(entries)})")

    with open(metadata_cache_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print("Done fetching metadata!")

if __name__ == '__main__':
    fetch_all_metadata()
