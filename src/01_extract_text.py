"""
Step 1: Extract clean text from all 787 FOMC PDFs.
"""
import pdfplumber
import pandas as pd
import re
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "create_sentiments" / "FOMC_Texts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "texts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []
failed = []

pdf_files = sorted(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF files")

for i, pdf_path in enumerate(pdf_files):
    filename = pdf_path.name
    print(f"[{i+1}/{len(pdf_files)}] Processing {filename}...", end=" ", flush=True)

    try:
        # Parse date and doc_type from filename: YYYY_MM_DD_doctype.pdf
        stem = pdf_path.stem
        parts = stem.split("_")
        meeting_date = f"{parts[0]}-{parts[1]}-{parts[2]}"
        doc_type = "_".join(parts[3:]) if len(parts) > 3 else "unknown"

        full_text = []
        page_count = 0
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Clean: normalize whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    # Remove page numbers (standalone digits at top/bottom)
                    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
                    full_text.append(text)
                page_count += 1

        clean_text = "\n\n".join(full_text)
        word_count = len(clean_text.split())

        # Save individual .txt file
        txt_filename = f"{meeting_date}_{doc_type}.txt"
        txt_path = OUTPUT_DIR / txt_filename
        txt_path.write_text(clean_text, encoding="utf-8")

        results.append({
            "meeting_date": meeting_date,
            "filename": filename,
            "doc_type": doc_type,
            "page_count": page_count,
            "word_count": word_count,
            "txt_path": str(txt_path)
        })
        print(f"OK ({page_count}pp, {word_count} words)")

    except Exception as e:
        failed.append({"filename": filename, "error": str(e)})
        print(f"FAILED: {e}")

# Save index
df_index = pd.DataFrame(results)
index_path = PROJECT_ROOT / "data" / "processed" / "text_index.csv"
df_index.to_csv(index_path, index=False)
print(f"\nIndex saved to {index_path}")

# Generate report
report = f"""# Step 1: Text Extraction Report

## Summary
- **Total files processed**: {len(results)} / {len(pdf_files)}
- **Total files failed**: {len(failed)}
- **Average word count**: {df_index['word_count'].mean():.0f}
- **Median word count**: {df_index['word_count'].median():.0f}
- **Date range**: {df_index['meeting_date'].min()} to {df_index['meeting_date'].max()}

## Document Type Distribution
{df_index['doc_type'].value_counts().to_string()}

## Files with Zero or Very Low Word Count (< 50)
{df_index[df_index['word_count'] < 50][['filename', 'word_count']].to_string()}

## Failed Files
{chr(10).join(f'- {f["filename"]}: {f["error"]}' for f in failed) if failed else 'None'}
"""

report_path = PROJECT_ROOT / "results" / "reports" / "01_extraction_report.md"
report_path.write_text(report)
print(f"Report saved to {report_path}")
print("Step 1 complete.")
