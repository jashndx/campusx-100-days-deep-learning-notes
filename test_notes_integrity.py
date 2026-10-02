import os
import re
import ast

NOTES_DIR = r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes'
SLIDES_DIR = os.path.join(NOTES_DIR, 'slides')

def test_notes_integrity():
    print("Testing Deep Learning 100 Exam Notes Integrity...")
    
    # 1. Check lecture count
    lecture_files = [f for f in os.listdir(NOTES_DIR) if f.startswith('Lecture_') and f.endswith('.md')]
    print(f"[CHECK] Individual lecture files count: {len(lecture_files)}")
    assert len(lecture_files) == 84, f"Expected 84 lectures, found {len(lecture_files)}"
    
    # 2. Check consolidated master
    consolidated_file = os.path.join(NOTES_DIR, 'CONSOLIDATED_MASTER_EXAM_NOTES.md')
    assert os.path.exists(consolidated_file), "CONSOLIDATED_MASTER_EXAM_NOTES.md missing!"
    size_bytes = os.path.getsize(consolidated_file)
    print(f"[CHECK] Consolidated file size: {size_bytes / 1024:.2f} KB")
    assert size_bytes > 500000, f"Consolidated file should be comprehensive (>500KB), got {size_bytes}"
    
    # 3. Check section contents across all lectures
    required_sections = [
        "1. Executive Summary & Core Intuition",
        "2. Key Definitions & Formal Terminology",
        "3. Mathematical Formulations & Derivations",
        "4. Architecture, Flowchart & Step-by-Step Algorithm",
        "6. High-Yield Exam & Viva Questions with Answers",
        "8. References, Notebooks & Supplementary Materials"
    ]
    
    all_md_files = lecture_files + ['CONSOLIDATED_MASTER_EXAM_NOTES.md']
    total_code_blocks = 0
    
    for lf in lecture_files:
        path = os.path.join(NOTES_DIR, lf)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        for sec in required_sections:
            assert sec in content, f"Missing section '{sec}' in {lf}"

    print(f"[PASS] All 84 lectures verified with complete structured sections!")
    
    # 4. Rigorous LaTeX and control character corruption audit
    corrupted_count = 0
    for mf in all_md_files:
        path = os.path.join(NOTES_DIR, mf)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Check forbidden control chars from unescaped LaTeX
        for bad in ['\x0c', '\x08', '\x07']:
            if bad in text:
                corrupted_count += text.count(bad)
                print(f"[ERROR] Found ASCII {ord(bad)} in {mf}")
        for bad in ['\times', '\text', '\tau', '\theta', '\top']:
            if bad in text:
                corrupted_count += text.count(bad)
                print(f"[ERROR] Found corrupted tab sequence {repr(bad)} in {mf}")
        for bad in ['\right', '\rho']:
            if bad in text:
                corrupted_count += text.count(bad)
                print(f"[ERROR] Found corrupted CR sequence {repr(bad)} in {mf}")
                
    assert corrupted_count == 0, f"Found {corrupted_count} corrupted LaTeX escapes across notes!"
    print("[PASS] LaTeX formulas verified: 0 corrupted escapes or control characters across all notes!")

    # 5. Check Python code block validity
    for mf in all_md_files:
        path = os.path.join(NOTES_DIR, mf)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
        for i, b in enumerate(blocks):
            total_code_blocks += 1
            b_clean = b.strip()
            try:
                ast.parse(b_clean)
            except SyntaxError as e:
                raise AssertionError(f"Syntax error in code block {i} of {mf}: {e}")
                
    print(f"[PASS] All {total_code_blocks} Python code blocks syntactically verified!")

    # 6. Check slides and research papers directory
    slide_files = os.listdir(SLIDES_DIR)
    print(f"[CHECK] Files in slides directory: {slide_files}")
    assert any('Transformers' in f for f in slide_files), "Transformers notes missing in slides!"
    assert any('Dropout' in f for f in slide_files), "Dropout paper missing in slides!"
    
    # Check research papers
    papers_dir = os.path.join(SLIDES_DIR, 'research_papers_and_readings')
    assert os.path.exists(papers_dir), "research_papers_and_readings directory missing!"
    papers = os.listdir(papers_dir)
    print(f"[CHECK] Seminal research papers count: {len(papers)}")
    assert len(papers) >= 6, f"Expected at least 6 seminal papers, found {len(papers)}"
    
    # Check notebooks
    notebooks_dir = os.path.join(SLIDES_DIR, 'notebooks_and_code')
    assert os.path.exists(notebooks_dir), "notebooks_and_code missing in slides!"
    nb_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]
    print(f"[CHECK] Downloaded official Colab notebooks count: {len(nb_files)}")
    assert len(nb_files) >= 20, f"Expected at least 20 notebooks, found {len(nb_files)}"
    
    print("\nALL VERIFICATION CHECKS PASSED PERFECTLY (100% PASS RATE)!")

if __name__ == '__main__':
    test_notes_integrity()
