import os
import requests

papers_dir = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\slides\research_papers_and_readings'
os.makedirs(papers_dir, exist_ok=True)

papers = [
    ("Attention_Is_All_You_Need_Vaswani2017.pdf", "https://arxiv.org/pdf/1706.03762.pdf"),
    ("Adam_Method_For_Stochastic_Optimization_Kingma2014.pdf", "https://arxiv.org/pdf/1412.6980.pdf"),
    ("Batch_Normalization_Accelerating_Deep_Network_Training_Ioffe2015.pdf", "https://arxiv.org/pdf/1502.03167.pdf"),
    ("Deep_Residual_Learning_for_Image_Recognition_He2015.pdf", "https://arxiv.org/pdf/1512.03385.pdf"),
    ("Layer_Normalization_Ba2016.pdf", "https://arxiv.org/pdf/1607.06450.pdf"),
    ("Bahdanau_Attention_Neural_Machine_Translation_2014.pdf", "https://arxiv.org/pdf/1409.0473.pdf"),
]

headers = {'User-Agent': 'Mozilla/5.0'}

for fname, url in papers:
    fpath = os.path.join(papers_dir, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 10000:
        print(f"[EXISTS] {fname}")
        continue
    print(f"Downloading {fname}...")
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(fpath, 'wb') as f:
                f.write(r.content)
            print(f"[SAVED] {fname} ({len(r.content)} bytes)")
        else:
            print(f"[FAIL] {fname}: status {r.status_code}")
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")

print("Papers download complete.")
