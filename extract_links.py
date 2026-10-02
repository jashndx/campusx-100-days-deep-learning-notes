import json
import re

with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\videos_metadata_raw.json', encoding='utf-8') as f:
    meta = json.load(f)

links_found = []
for vid, data in meta.items():
    desc = data.get('description', '')
    title = data.get('title', '')
    idx = data.get('index', 0)
    urls = re.findall(r'https?://[^\s<>"]+', desc)
    # Filter for slides, code, colab, github, drive
    interesting = []
    for u in urls:
        u_clean = u.rstrip('.,;:)')
        if any(k in u_clean.lower() for k in ['github', 'drive.google', 'colab', 'slide', 'notes', 'docs.google', 'campusx', 'kaggle']):
            interesting.append(u_clean)
    if interesting:
        links_found.append({
            'index': idx,
            'title': title,
            'id': vid,
            'links': interesting
        })

print(f"Total videos with interesting links: {len(links_found)}")
for item in links_found[:15]:
    print(f"Video {item['index']:02d}: {item['title'][:40]}")
    for l in item['links']:
        print(f"   {l}")

with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\all_extracted_links.json', 'w', encoding='utf-8') as out:
    json.dump(links_found, out, indent=2)
