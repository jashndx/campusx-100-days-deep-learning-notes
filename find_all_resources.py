import json
import re

with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\videos_metadata_raw.json', encoding='utf-8') as f:
    meta = json.load(f)

print(f"Total videos: {len(meta)}")
resources = {}

for vid, data in meta.items():
    idx = data.get('index', 0)
    title = data.get('title', '')
    desc = data.get('description', '')
    
    # Extract links
    urls = re.findall(r'https?://[^\s<>"]+', desc)
    github_links = [u for u in urls if 'github.com' in u]
    colab_links = [u for u in urls if 'colab.research.google.com' in u]
    kaggle_links = [u for u in urls if 'kaggle.com' in u]
    drive_links = [u for u in urls if 'drive.google.com' in u or 'docs.google.com' in u]
    notes_links = [u for u in urls if 'learnwith.campusx.in' in u or 'notes' in u.lower() or 'slide' in u.lower() or 'pdf' in u.lower()]
    
    if github_links or colab_links or kaggle_links or drive_links:
        resources[idx] = {
            'title': title,
            'github': github_links,
            'colab': colab_links,
            'kaggle': kaggle_links,
            'drive': drive_links,
            'notes': notes_links
        }

print(f"Videos with external code/slides/colab/drive resources: {len(resources)}")
for idx, res in sorted(resources.items()):
    print(f"\nLecture {idx:02d}: {res['title']}")
    for k, v in res.items():
        if k != 'title' and v:
            print(f"  {k}: {v}")
