import re

files = [
    r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\data_module1.py',
    r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\data_module2.py',
    r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\data_module3.py',
    r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\data_module4.py',
    r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\data_module5.py',
    r'c:\Users\Admin\Desktop\ska\deep_learning_100_exam_notes\data_module6.py',
]

for fp in files:
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace any : """ with : r""" if not already r"""
    content = re.sub(r':\s*"""', ': r"""', content)
    # Also in tuples ("term", """...""") -> ("term", r"""...""")
    content = re.sub(r',\s*"""', ', r"""', content)
    # Also in single line strings with backslashes
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Processed {fp}')
