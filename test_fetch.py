import json
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

def test_fetch(video_id):
    print(f"Testing video {video_id}...")
    # Test transcript
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)
        print("Available transcripts:", [t.language_code for t in transcript_list])
        transcript = transcript_list.find_transcript(['en', 'hi', 'en-IN', 'hi-Latn']).fetch()
        text = " ".join([item['text'] for item in transcript[:10]])
        print(f"Transcript sample ({len(transcript)} segments): {text[:200]}...")
    except Exception as e:
        print(f"Transcript error: {e}")

    # Test description & metadata via yt-dlp
    try:
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            desc = info.get('description', '')
            print(f"Description length: {len(desc)}")
            print("Description snippet:\n", "\n".join(desc.splitlines()[:10]))
    except Exception as e:
        print(f"yt-dlp error: {e}")

if __name__ == '__main__':
    test_fetch("2dH_qjc9mFg") # Video 1
    print("="*50)
    test_fetch("fHF22Wxuyw4") # Video 2
    print("="*50)
    test_fetch("6M1wWQmcUjQ") # Video 15
