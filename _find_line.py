from config import PAPER_DIR
lines = open(os.path.join(PAPER_DIR, 'paper_jne.tex'), encoding='utf-8').readlines()
for i, l in enumerate(lines):
    if '\\section' in l or '\\subsection' in l:
        print(f"L{i+1}: {l.rstrip()[:120]}")