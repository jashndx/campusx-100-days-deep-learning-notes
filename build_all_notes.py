import os
import json
from generator_base import (
    get_metadata, get_transcript, clean_slug, render_lecture_note, OUTPUT_DIR
)
from data_module1 import MODULE_1_LECTURES
from data_module2 import MODULE_2_LECTURES
from data_module3 import MODULE_3_LECTURES
from data_module4 import MODULE_4_LECTURES
from data_module5 import MODULE_5_LECTURES
from data_module6 import MODULE_6_LECTURES

ALL_MODULES = [
    ("Module 1: Foundations of Deep Learning & Multi-Layer Perceptrons", MODULE_1_LECTURES),
    ("Module 2: Backpropagation, Optimization & Regularization", MODULE_2_LECTURES),
    ("Module 3: Advanced Training, Activations, Initializations & Modern Optimizers", MODULE_3_LECTURES),
    ("Module 4: Convolutional Neural Networks (CNNs) & Computer Vision", MODULE_4_LECTURES),
    ("Module 5: Recurrent Neural Networks (RNNs), LSTMs, GRUs & Language Evolution", MODULE_5_LECTURES),
    ("Module 6: Sequence-to-Sequence, Attention Mechanisms & Complete Transformer Architecture", MODULE_6_LECTURES),
]

def main():
    metadata = get_metadata()
    all_lectures = []
    for mod_name, mod_list in ALL_MODULES:
        for lec in mod_list:
            all_lectures.append(lec)

    print(f"Total lectures to process: {len(all_lectures)}")
    
    generated_files = []
    
    # 1. Generate Individual Lecture Notes
    for lec in all_lectures:
        idx = lec['index']
        title = lec['title']
        slug = clean_slug(title)
        filename = f"Lecture_{idx:03d}_{slug}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Render markdown
        content = render_lecture_note(metadata, lec)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        generated_files.append((filename, len(content)))
        print(f"[{idx:02d}/84] Generated: {filename} ({len(content)} bytes)")
        
    print(f"Successfully generated all {len(generated_files)} individual lecture notes!")

    # 2. Build CONSOLIDATED_MASTER_EXAM_NOTES.md
    print("Compiling CONSOLIDATED_MASTER_EXAM_NOTES.md...")
    consolidated_path = os.path.join(OUTPUT_DIR, "CONSOLIDATED_MASTER_EXAM_NOTES.md")
    
    master = []
    master.append("# 100 Days of Deep Learning: Master Comprehensive Exam Study Guide")
    master.append("\n> **Comprehensive University & Viva Exam Preparation Compendium**")
    master.append("> Based on CampusX's Complete 84-Lecture Deep Learning Curriculum by Nitish Singh.")
    master.append("> Covers complete theory, rigorous mathematical derivations, tensor dimensional analysis, code implementations, and high-yield viva questions across all modules.\n")
    master.append("---\n")
    
    # Master Table of Contents
    master.append("## Table of Contents & Syllabus Map\n")
    for mod_name, mod_list in ALL_MODULES:
        master.append(f"### {mod_name}")
        for lec in mod_list:
            idx = lec['index']
            title = lec['title']
            slug = clean_slug(title)
            filename = f"Lecture_{idx:03d}_{slug}.md"
            master.append(f"- [Lecture {idx:03d}: {title}](#lecture-{idx:03d}-{slug.lower().replace('_', '-')}) ([Standalone File]({filename}))")
        master.append("")
        
    master.append("---\n")
    
    # High-Yield Formula Sheet
    master.append("## High-Yield Mathematical Formula & Equation Cheat Sheet\n")
    master.append("""
### 1. Feedforward & Activations
- **Affine Linear Transformation:** $\\mathbf{Z}^{[l]} = \\mathbf{A}^{[l-1]} \\mathbf{W}^{[l]} + \\mathbf{b}^{[l]}$
- **Sigmoid:** $\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$
- **Tanh:** $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}, \\quad \\tanh'(z) = 1 - \\tanh^2(z)$
- **ReLU:** $f(z) = \\max(0, z), \\quad f'(z) = 1 \\text{ for } z > 0$
- **Softmax:** $\\hat{y}_k = \\frac{e^{z_k}}{\\sum_{j=1}^K e^{z_j}}$

### 2. Loss Functions
- **Mean Squared Error (MSE):** $\\mathcal{L} = \\frac{1}{N} \\sum (y - \\hat{y})^2$
- **Binary Cross-Entropy (BCE):** $\\mathcal{L} = -\\frac{1}{N} \\sum [y \\log \\hat{y} + (1-y)\\log(1-\\hat{y})]$
- **Categorical Cross-Entropy (CCE):** $\\mathcal{L} = -\\frac{1}{N} \\sum_{i} \\sum_{k} y_{ik} \\log \\hat{y}_{ik}$

### 3. Backpropagation (The 4 Fundamental Equations)
- **Output Error:** $\\boldsymbol{\\delta}^{[L]} = \\nabla_{\\mathbf{a}} \\mathcal{L} \\odot g'(\\mathbf{z}^{[L]})$ (For BCE+Sigmoid or CCE+Softmax: $\\boldsymbol{\\delta}^{[L]} = \\hat{\\mathbf{y}} - \\mathbf{y}$)
- **Hidden Error:** $\\boldsymbol{\\delta}^{[l]} = ((\\mathbf{W}^{[l+1]})^T \\boldsymbol{\\delta}^{[l+1]}) \\odot g'(\\mathbf{z}^{[l]})$
- **Weight Gradient:** $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}^{[l]}} = \\frac{1}{M} (\\mathbf{A}^{[l-1]})^T \\boldsymbol{\\delta}^{[l]}$
- **Bias Gradient:** $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{b}^{[l]}} = \\frac{1}{M} \\sum \\boldsymbol{\\delta}^{[l]}$

### 4. Regularization & Optimization
- **$L2$ Weight Decay Update:** $\\mathbf{W} \\leftarrow \\left(1 - \\frac{\\eta \\lambda}{m}\\right)\\mathbf{W} - \\eta \\nabla \\mathcal{L}$
- **Inverted Dropout:** $\\widetilde{\\mathbf{a}} = \\frac{\\mathbf{r} \\odot \\mathbf{a}}{1 - p}$
- **He Normal Initialization:** $\\sigma = \\sqrt{\\frac{2}{n_{in}}}$
- **Xavier Normal Initialization:** $\\sigma = \\sqrt{\\frac{2}{n_{in} + n_{out}}}$
- **Batch Normalization:** $\\hat{z} = \\frac{z - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}, \\quad y = \\gamma \\hat{z} + \\beta$
- **SGD with Momentum:** $\\mathbf{v}_t = \\beta \\mathbf{v}_{t-1} + \\eta \\mathbf{g}_t, \\quad \\mathbf{w} \\leftarrow \\mathbf{w} - \\mathbf{v}_t$
- **RMSProp:** $\\mathbf{v}_t = \\beta \\mathbf{v}_{t-1} + (1-\\beta)\\mathbf{g}_t^2, \\quad \\mathbf{w} \\leftarrow \\mathbf{w} - \\frac{\\eta}{\\sqrt{\\mathbf{v}_t} + \\epsilon} \\mathbf{g}_t$
- **Adam:** $\\hat{\\mathbf{m}}_t = \\frac{\\mathbf{m}_t}{1 - \\beta_1^t}, \\ \\hat{\\mathbf{v}}_t = \\frac{\\mathbf{v}_t}{1 - \\beta_2^t}, \\quad \\mathbf{w} \\leftarrow \\mathbf{w} - \\frac{\\eta}{\\sqrt{\\hat{\\mathbf{v}}_t} + \\epsilon} \\hat{\\mathbf{m}}_t$

### 5. Convolutional Networks (CNNs)
- **Output Spatial Dimension:** $M = \\left\\lfloor \\frac{N - K + 2P}{S} \\right\\rfloor + 1$
- **Same Padding:** $P = \\frac{K - 1}{2}$
- **Conv Parameters:** $(K_h \\times K_w \\times C_{in} + 1) \\times C_{out}$

### 6. Recurrent Networks (RNNs, LSTMs, GRUs)
- **Vanilla RNN:** $\\mathbf{h}_t = \\tanh(\\mathbf{W}_{hh} \\mathbf{h}_{t-1} + \\mathbf{W}_{xh} \\mathbf{x}_t + \\mathbf{b}_h)$
- **LSTM Gates:** $\\mathbf{f}_t = \\sigma(\\dots), \\ \\mathbf{i}_t = \\sigma(\\dots), \\ \\widetilde{\\mathbf{C}}_t = \\tanh(\\dots), \\ \\mathbf{o}_t = \\sigma(\\dots)$
- **LSTM Cell Update:** $\\mathbf{C}_t = \\mathbf{f}_t \\odot \\mathbf{C}_{t-1} + \\mathbf{i}_t \\odot \\widetilde{\\mathbf{C}}_t, \\quad \\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{C}_t)$
- **LSTM Parameters:** $4 \\times [h(d + h + 1)]$
- **GRU Parameters:** $3 \\times [h(d + h + 1)]$

### 7. Attention & Transformers
- **Scaled Dot-Product Attention:** $\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{Softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right)\\mathbf{V}$
- **Multi-Head Attention:** $\\text{MultiHead}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) \\mathbf{W}_O$
- **Sinusoidal Positional Encoding:** $PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right), \\ PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)$
- **Layer Normalization:** $\\hat{x} = \\frac{x - \\mu_{layer}}{\\sqrt{\\sigma_{layer}^2 + \\epsilon}}, \\quad y = \\gamma \\hat{x} + \\beta$
- **FFN:** $\\text{FFN}(x) = \\max(0, x\\mathbf{W}_1 + \\mathbf{b}_1)\\mathbf{W}_2 + \\mathbf{b}_2$
""")
    master.append("\n---\n")

    # Consolidate all lectures
    for mod_name, mod_list in ALL_MODULES:
        master.append(f"# {mod_name.upper()}\n")
        for lec in mod_list:
            content = render_lecture_note(metadata, lec)
            master.append(content)
            master.append("\n\n---\n\n")

    full_master_text = "\n".join(master)
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        f.write(full_master_text)
        
    print(f"CONSOLIDATED_MASTER_EXAM_NOTES.md successfully generated! Total size: {len(full_master_text)} characters ({len(full_master_text)/1024:.2f} KB).")

if __name__ == '__main__':
    main()
