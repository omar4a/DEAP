import PyPDF2
import os
import re

pdf_dir = r"c:\Omar\Education\GP\DEAP"
pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

patterns = [
    r"(.{0,200}\d{1,2}[\-\s]fold.{0,200})",
    r"(.{0,200}cross[\-\s]validation.{0,200})",
    r"(.{0,200}leave[\-\s]one[\-\s]subject[\-\s]out.{0,200})",
    r"(.{0,200}overlap.{0,200})",
    r"(.{0,200}split.{0,200})"
]

for pdf in pdfs:
    path = os.path.join(pdf_dir, pdf)
    
    try:
        reader = PyPDF2.PdfReader(path)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: full_text += t.replace("\n", " ")
        
        full_text_lower = full_text.lower()
        if "eegnet" in full_text_lower or "accuracy" in full_text_lower:
            print(f"\n======== {pdf} ========")
            found = False
            for p in patterns:
                matches = re.findall(p, full_text_lower)
                if matches:
                    found = True
                    for m in matches[:2]: # First two matches per pattern
                        print(f" -> {m}")
    except Exception as e:
        pass
