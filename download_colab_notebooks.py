import json
import os
import re
import urllib.request
import time

slides_dir = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\slides\notebooks_and_code'
os.makedirs(slides_dir, exist_ok=True)

with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\all_extracted_links.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

downloaded = 0
failed = 0

for item in data:
    idx = item['index']
    title = item['title']
    clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    for link in item.get('links', []):
        file_id = None
        if 'colab.research.google.com/drive/' in link:
            match = re.search(r'/drive/([a-zA-Z0-9_-]+)', link)
            if match:
                file_id = match.group(1)
        elif 'drive.google.com/file/d/' in link:
            match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', link)
            if match:
                file_id = match.group(1)

        if file_id:
            # Try to download
            export_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            out_filename = f"Lecture_{idx:03d}_{clean_title[:30]}_{file_id[:8]}.ipynb"
            out_path = os.path.join(slides_dir, out_filename)
            if os.path.exists(out_path):
                print(f"[EXISTS] {out_filename}")
                continue

            try:
                req = urllib.request.Request(export_url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req, timeout=15)
                content = resp.read()
                if len(content) > 500:
                    with open(out_path, 'wb') as out_f:
                        out_f.write(content)
                    downloaded += 1
                    print(f"[SAVED] {out_filename} ({len(content)} bytes)")
                else:
                    failed += 1
                    print(f"[SKIPPED] {out_filename} (Too small: {len(content)} bytes)")
            except Exception as e:
                failed += 1
                print(f"[FAIL] {out_filename} ({link}): {e}")
            time.sleep(0.5)

print(f"Finished downloading notebooks. Downloaded: {downloaded}, Failed: {failed}")
