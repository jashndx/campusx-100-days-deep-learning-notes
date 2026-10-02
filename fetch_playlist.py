import json
import yt_dlp

def main():
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
    }
    url = 'https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn'
    print(f"Fetching playlist from {url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title')
        entries = list(info.get('entries', []))
        print(f"Playlist Title: {title}")
        print(f"Total videos: {len(entries)}")
        
        with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\playlist_summary.json', 'w', encoding='utf-8') as f:
            json.dump({'title': title, 'count': len(entries), 'entries': entries}, f, indent=2, ensure_ascii=False)
        print("Saved to playlist_summary.json")

if __name__ == '__main__':
    main()
