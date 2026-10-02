import json

with open(r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\playlist_summary.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total entries: {data['count']}")
for i, e in enumerate(data['entries']):
    print(f"{i+1:02d}: {e.get('id')} | {e.get('title')}")
