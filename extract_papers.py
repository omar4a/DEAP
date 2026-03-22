import PyPDF2
import os
import re

pdf_dir = r"c:\Omar\Education\GP\DEAP"
pdfs = [
    "8. Enhanced Cross-Task EEG Classification,  Domain Adaptation with EEGNet.pdf",
    "9. Deep learning-based EEG emotion recognition, a comprehensive review.pdf",
    "12. Enhancing EEG-Based Emotion Detection with Hybrid Models, Insights from DEAP Dataset Applications.pdf",
    "14. EEG Emotion Classification based on Valence and Arousal using DEAP dataset.pdf"
]

keywords = ["EEGNet", "cross-validation", "window", "leakage", "accuracy", "fold", "split", "trial"]

for pdf in pdfs:
    path = os.path.join(pdf_dir, pdf)
    if not os.path.exists(path):
        continue
    
    print(f"\n--- Analyzing {pdf} ---")
    try:
        reader = PyPDF2.PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            
            # Look for paragraphs containing our keywords
            paragraphs = text.split('\n\n')
            for p in paragraphs:
                p_lower = p.lower()
                if "eegnet" in p_lower and ("accuracy" in p_lower or "cross-validation" in p_lower or "split" in p_lower):
                    print(f"[Page {i+1}] {p[:300]}...\n")
                elif "10-fold" in p_lower or "5-fold" in p_lower or "leave-one-subject-out" in p_lower:
                    print(f"[Page {i+1}] {p[:300]}...\n")
    except Exception as e:
        print(f"Error reading {pdf}: {e}")
