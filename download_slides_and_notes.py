import os
import shutil
import requests

slides_dir = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\slides'
os.makedirs(slides_dir, exist_ok=True)

# 1. Transformers Google Doc Notes (PDF & TXT)
doc_id = '1OsMe9jGHoZS67FH8TdIzcUaDWuu5RAbCbBKk2cNq6Dk'
pdf_url = f'https://docs.google.com/document/d/{doc_id}/export?format=pdf'
txt_url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'

headers = {'User-Agent': 'Mozilla/5.0'}

print("Downloading Transformers Official Course Document PDF...")
try:
    r = requests.get(pdf_url, headers=headers, timeout=30)
    if r.status_code == 200 and len(r.content) > 1000:
        pdf_path = os.path.join(slides_dir, 'Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved: {pdf_path} ({len(r.content)} bytes)")
    else:
        print(f"Failed PDF download: status {r.status_code}")
except Exception as e:
    print(f"Error downloading PDF: {e}")

print("Downloading Transformers Official Course Document TXT...")
try:
    r = requests.get(txt_url, headers=headers, timeout=30)
    if r.status_code == 200 and len(r.content) > 1000:
        txt_path = os.path.join(slides_dir, 'Transformers_Complete_Course_Notes_Lectures_78_to_84.txt')
        with open(txt_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved: {txt_path} ({len(r.content)} bytes)")
    else:
        print(f"Failed TXT download: status {r.status_code}")
except Exception as e:
    print(f"Error downloading TXT: {e}")

# 2. Dropout Seminal Research Paper (Srivastava et al., JMLR)
dropout_url = 'https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf'
print("Downloading Dropout Paper PDF...")
try:
    r = requests.get(dropout_url, headers=headers, timeout=30)
    if r.status_code == 200 and len(r.content) > 1000:
        paper_path = os.path.join(slides_dir, 'Dropout_A_Simple_Way_to_Prevent_Overfitting_Srivastava2014.pdf')
        with open(paper_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved: {paper_path} ({len(r.content)} bytes)")
    else:
        print(f"Failed Dropout paper download: status {r.status_code}")
except Exception as e:
    print(f"Error downloading Dropout paper: {e}")

# 3. Copy official repo days into slides/notebooks
repo_dir = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\official_repo'
notebooks_dir = os.path.join(slides_dir, 'notebooks_and_code')
os.makedirs(notebooks_dir, exist_ok=True)
if os.path.exists(repo_dir):
    for item in os.listdir(repo_dir):
        if item.startswith('day'):
            src = os.path.join(repo_dir, item)
            dst = os.path.join(notebooks_dir, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"Copied {item} to {notebooks_dir}")

print("Slides and materials download finished!")
