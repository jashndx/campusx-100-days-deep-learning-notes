import sys
from youtube_transcript_api import YouTubeTranscriptApi

try:
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list('fHF22Wxuyw4')
    for t in transcript_list:
        print(f"Transcript language: {t.language_code}, is_generated: {t.is_generated}")
    
    # fetch english or hindi
    tr = transcript_list.find_transcript(['en', 'en-US', 'hi']).fetch()
    print("Type of item:", type(tr[0]))
    print("Dir:", [d for d in dir(tr[0]) if not d.startswith('_')])
    print("Text attribute:", getattr(tr[0], 'text', None))
    full_text = " ".join([getattr(item, 'text', str(item)) for item in tr[:10]])
    print("Sample text:", full_text)
except Exception as e:
    print("Error:", e)
