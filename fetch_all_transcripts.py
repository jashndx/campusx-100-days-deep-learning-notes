import json
import os
import time
from youtube_transcript_api import YouTubeTranscriptApi

def fetch_all_transcripts():
    with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\playlist_summary.json', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = data['entries']
    transcripts_dir = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\transcripts'
    os.makedirs(transcripts_dir, exist_ok=True)
    
    ytt = YouTubeTranscriptApi()
    
    summary = {}
    summary_path = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\transcripts_summary.json'
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

    for idx, entry in enumerate(entries):
        vid = entry['id']
        title = entry.get('title', '')
        out_file = os.path.join(transcripts_dir, f"{idx+1:03d}_{vid}.json")
        
        if os.path.exists(out_file) and vid in summary and summary[vid].get('status') == 'success':
            continue
            
        print(f"[{idx+1}/{len(entries)}] Fetching transcript for {vid}: {title[:35]}...")
        try:
            transcript_list = ytt.list(vid)
            # Find english or hindi or auto-generated
            available_langs = [t.language_code for t in transcript_list]
            
            # Prefer English, then Hindi, then any
            chosen_transcript = None
            try:
                chosen_transcript = transcript_list.find_transcript(['en', 'en-US', 'en-IN'])
            except Exception:
                try:
                    chosen_transcript = transcript_list.find_transcript(['hi', 'hi-Latn'])
                except Exception:
                    # pick the first available
                    for t in transcript_list:
                        chosen_transcript = t
                        break
                        
            if chosen_transcript:
                data_snippets = chosen_transcript.fetch()
                raw_text_list = []
                full_text_parts = []
                for snip in data_snippets:
                    t_text = getattr(snip, 'text', '')
                    start = getattr(snip, 'start', 0.0)
                    duration = getattr(snip, 'duration', 0.0)
                    raw_text_list.append({'text': t_text, 'start': start, 'duration': duration})
                    full_text_parts.append(t_text)
                
                full_text = " ".join(full_text_parts)
                with open(out_file, 'w', encoding='utf-8') as out_f:
                    json.dump({
                        'id': vid,
                        'index': idx + 1,
                        'title': title,
                        'language': chosen_transcript.language_code,
                        'is_generated': chosen_transcript.is_generated,
                        'full_text': full_text,
                        'snippets': raw_text_list
                    }, out_f, indent=2, ensure_ascii=False)
                    
                summary[vid] = {
                    'index': idx + 1,
                    'status': 'success',
                    'language': chosen_transcript.language_code,
                    'length_chars': len(full_text),
                    'word_count': len(full_text.split())
                }
                print(f"  -> Success ({chosen_transcript.language_code}, {len(full_text.split())} words)")
            else:
                summary[vid] = {'index': idx + 1, 'status': 'no_transcript'}
                print("  -> No transcript found")
        except Exception as e:
            print(f"  -> Error: {e}")
            summary[vid] = {'index': idx + 1, 'status': 'error', 'error': str(e)}
            
        time.sleep(0.5)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Done fetching transcripts!")

if __name__ == '__main__':
    fetch_all_transcripts()
