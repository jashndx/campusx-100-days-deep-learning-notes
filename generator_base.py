import json
import os
import re

TRANSCRIPTS_DIR = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\transcripts'
METADATA_FILE = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\videos_metadata_raw.json'
OUTPUT_DIR = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes'

def get_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_transcript(video_id):
    for fname in os.listdir(TRANSCRIPTS_DIR):
        if video_id in fname and fname.endswith('.json'):
            path = os.path.join(TRANSCRIPTS_DIR, fname)
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

def clean_slug(title):
    t_clean = re.sub(r'\|.*$', '', title)
    t_clean = re.sub(r'[^\w\s-]', '', t_clean).strip()
    slug = re.sub(r'[-\s]+', '_', t_clean)
    return slug

def render_lecture_note(meta_dict, lecture_data):
    """
    lecture_data is expected to have:
    - index: int
    - title: str
    - video_id: str
    - duration_str: str
    - core_intuition: str
    - key_definitions: list of (term, definition)
    - mathematical_formulations: str (LaTeX)
    - architecture_and_algorithm: str
    - code_snippet: str
    - exam_viva_qa: list of (question, answer)
    - resources_and_references: list of (name, url_or_desc)
    """
    idx = lecture_data['index']
    vid = lecture_data.get('video_id', '')
    title = lecture_data['title']
    duration_str = lecture_data.get('duration_str', 'N/A')
    
    # Check if we have transcript
    tr = get_transcript(vid)
    tr_stat = f"Available ({tr.get('language')}, {len(tr.get('full_text','').split())} words)" if tr else "Available via official notes & syllabus outline"
    
    md = []
    md.append(f"# Lecture {idx:03d}: {title}")
    md.append(f"\n> **CampusX 100 Days of Deep Learning** | Video ID: `{vid}` | Duration: {duration_str}")
    md.append(f"> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v={vid}) | **Transcript Status**: {tr_stat}\n")
    md.append("---\n")
    
    md.append("## 1. Executive Summary & Core Intuition")
    md.append(lecture_data['core_intuition'].strip())
    md.append("\n")
    
    md.append("## 2. Key Definitions & Formal Terminology")
    for term, defn in lecture_data['key_definitions']:
        md.append(f"- **{term}**: {defn}")
    md.append("\n")
    
    md.append("## 3. Mathematical Formulations & Derivations")
    md.append(lecture_data['mathematical_formulations'].strip())
    md.append("\n")
    
    md.append("## 4. Architecture, Flowchart & Step-by-Step Algorithm")
    md.append(lecture_data['architecture_and_algorithm'].strip())
    md.append("\n")
    
    if lecture_data.get('code_snippet'):
        md.append("## 5. Implementation Code Snippet")
        md.append(lecture_data['code_snippet'].strip())
        md.append("\n")
    
    md.append("## 6. High-Yield Exam & Viva Questions with Answers")
    for q_idx, (q, a) in enumerate(lecture_data['exam_viva_qa'], 1):
        md.append(f"### Q{q_idx}: {q}")
        md.append(f"**Answer**:\n{a}\n")
        
    md.append("## 7. Crucial Exam Takeaways & Common Pitfalls")
    if lecture_data.get('exam_takeaways'):
        for item in lecture_data['exam_takeaways']:
            md.append(f"- {item}")
    else:
        md.append("- Pay special attention to dimensional analysis and tensor shapes in exams.")
        md.append("- Memorize forward and backward update equations with precise indices.")
    md.append("\n")
    
    md.append("## 8. References, Notebooks & Supplementary Materials")
    md.append(f"- [CampusX Video Link](https://www.youtube.com/watch?v={vid})")
    for name, ref in lecture_data.get('resources_and_references', []):
        md.append(f"- [{name}]({ref})")
    md.append("\n---\n")
    
    return "\n".join(md)
