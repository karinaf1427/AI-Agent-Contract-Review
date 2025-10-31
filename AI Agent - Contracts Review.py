
#Model:
#all-MiniLM-L6-v2
#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

!pip install --quiet python-docx pdfplumber pytesseract pillow sentence-transformers pandas

import os
import re
import io
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import pdfplumber
import docx
from PIL import Image
import pytesseract
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from google.colab import files

MODEL_NAME = "all-MiniLM-L6-v2"  #developed by Hugging Face
print("Loading sentence-transformers model:", MODEL_NAME)
sbert = SentenceTransformer(MODEL_NAME)

# 4) Document Processing
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_chunks.append(txt)
            if not txt:
                try:
                    for im in page.images:

                        pass
                except Exception:
                    pass
    return "\n".join(text_chunks)

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def ingest_uploaded_file(filedict) -> str:
    texts = []
    for fname, content in filedict.items():
        lower = fname.lower()
        print("Ingesting:", fname)
        if lower.endswith(".pdf"):
            texts.append(extract_text_from_pdf_bytes(content))
        elif lower.endswith(".docx"):
            texts.append(extract_text_from_docx_bytes(content))
        elif lower.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            try:
                im = Image.open(io.BytesIO(content))
                ocr = pytesseract.image_to_string(im)
                texts.append(ocr)
            except Exception as e:
                print("Unsupported format or error reading file:", fname, e)
    return "\n\n".join(texts)


# 5) Clause segmentation (simple heuristic)
def split_into_clauses(text: str) -> List[str]:
    header_pattern = re.compile(r'(^\s*(\d+(\.\d+)*\s+|ARTICLE\s+\d+|[A-Z][A-Z ]{4,}):?)', re.MULTILINE)
    parts = header_pattern.split(text)
    if len(parts) < 3:
        segments = [seg.strip() for seg in re.split(r'\n{2,}', text) if seg.strip()]
    else:
        segments = []
        accum = ""
        for chunk in parts:
            if chunk is not None and chunk.strip().isupper() and len(chunk.strip()) > 4:
                if accum:
                    segments.append(accum.strip())
                accum = chunk
            elif chunk is not None:
                accum += " " + chunk
        if accum:
            segments.append(accum.strip())
        segments = [s for s in segments if len(s) > 40]
    final = []
    for s in segments:
        if len(s) < 200:
            final.append(s)
        else:
            sentences = re.split(r'(?<=[\.\?\!])\s+', s)
            for sent in sentences:
                if sent.strip():
                    final.append(sent.strip())
    return final


# 6) Clause type templates & policy DB (example, customize these)
CLAUSE_TEMPLATES = {
    "confidentiality": "Confidentiality: The parties agree that Confidential Information shall be kept confidential and shall not be disclosed to any third party except as required by law. The Receiving Party shall use at least the same degree of care as it uses to protect its own confidential information, but in no event less than reasonable care.",
    "termination": "Termination for convenience: Either party may terminate this Agreement upon thirty (30) days' prior written notice to the other party. Upon termination, all outstanding fees due through the effective date of termination shall be payable.",
    "liability_cap": "Limitation of liability: Except for liability arising from willful misconduct or gross negligence, each party's aggregate liability under this Agreement shall not exceed the total amount paid or payable by Customer under this Agreement in the twelve (12) months preceding the event giving rise to the claim.",
    "data_protection": "Data Protection: The parties will comply with applicable data protection laws. The Processor will implement appropriate technical and organizational measures to protect personal data. Cross-border transfers shall be governed by the appropriate legal mechanism.",
    "ip_assignment": "IP Assignment: All intellectual property created specifically in the performance of this Agreement by Contractor shall be deemed 'works made for hire' and is assigned to Company. Contractor hereby irrevocably assigns all right, title, and interest in such IP to Company."
}

# Policies (example)
POLICIES = {
    "required_clauses": ["confidentiality", "liability_cap", "data_protection"],
    "forbidden_phrases": ["unlimited liability", "no liability for willful misconduct"],
    "liability_cap_max_multiplier": 1.0
}

# 7) Clause classification: compute embeddings for templates once
template_texts = list(CLAUSE_TEMPLATES.values())
template_keys = list(CLAUSE_TEMPLATES.keys())
template_embeddings = sbert.encode(template_texts, convert_to_tensor=True)


def classify_clause_texts(clauses: List[str], top_k: int = 1) -> List[Tuple[str, float]]:
    """
    For each clause text, returns the best matching template key and similarity score.
    """
    clause_emb = sbert.encode(clauses, convert_to_tensor=True)
    hits = util.semantic_search(clause_emb, template_embeddings, top_k=top_k)
    results = []
    for idx, hit in enumerate(hits):
        best = hit[0]
        tmpl_idx = best['corpus_id']
        score = best['score']
        key = template_keys[tmpl_idx]
        results.append((key, float(score)))
    return results


# 8) Policy checks & suggestions
def check_policies(clauses: List[str], classified: List[Tuple[str, float]], contract_value: float = None) -> Dict[str, Any]:
    """
    Returns: {
      "found": {clause_type: [indices]},
      "missing_required": [clause_types],
      "forbidden_matches": [(clause_index, phrase, context)],
      "liability_issues": [(clause_index, reason)],
      "suggestions": {clause_type: suggestion_text}
    }
    """
    found = defaultdict(list)
    for i, (ctype, score) in enumerate(classified):
        if score > 0.45:  # threshold - tune as needed
            found[ctype].append(i)

    missing_required = [c for c in POLICIES["required_clauses"] if c not in found or not found[c]]

    forbidden_matches = []
    for idx, text in enumerate(clauses):
        for phrase in POLICIES["forbidden_phrases"]:
            if phrase.lower() in text.lower():
                forbidden_matches.append((idx, phrase, text[:400]))

    liability_issues = []
    for idx, text in enumerate(clauses):
        if "liabil" in text.lower() or "limitation of liability" in text.lower():
            if "unlimited" in text.lower():
                liability_issues.append((idx, "unlimited_liability"))
            else:
                m = re.search(r'not exceed (\$?[\d,\.]+)', text)
                if m:
                    val = m.group(1)
                    liability_issues.append((idx, f"cap_specified:{val}"))
                else:
                    liability_issues.append((idx, "no_cap_found"))
    suggestions = {}
    for req in missing_required:
        suggestions[req] = CLAUSE_TEMPLATES.get(req, f"Suggested clause for {req}: [Please provide clause].")

    for (idx, phrase, context) in forbidden_matches:
        suggestions[f"forbidden_{idx}"] = f"Found forbidden phrase '{phrase}' in clause {idx}. Suggest replacing with safer language (e.g., limited liability with cap)."

    for (idx, issue) in liability_issues:
        if issue == "unlimited_liability":
            suggestions[f"liability_{idx}"] = "Clause appears to create unlimited liability. Suggest adding a monetary cap tied to contract value or fees paid in the prior 12 months."
        elif issue.startswith("cap_specified"):
            suggestions[f"liability_{idx}"] = f"Clause specifies cap {issue.split(':',1)[1]}. Confirm if that complies with company policy."
        elif issue == "no_cap_found":
            suggestions[f"liability_{idx}"] = "No explicit liability cap found. Suggest adding a limitation of liability clause consistent with policy."

    if contract_value:
        pass

    return {
        "found": dict(found),
        "missing_required": missing_required,
        "forbidden_matches": forbidden_matches,
        "liability_issues": liability_issues,
        "suggestions": suggestions
    }

# 10) High-level pipeline
def review_contract_text(text: str, contract_value: float = None, use_openai_for_rewrite: bool = False) -> Dict[str, Any]:
    clauses = split_into_clauses(text)
    classification = classify_clause_texts(clauses)
    policy_report = check_policies(clauses, classification, contract_value=contract_value)

    # generate suggested redlines for problematic clauses
    redlines = {}
    # suggestions for each flagged clause
    for key, val in policy_report["suggestions"].items():
        if key.startswith("forbidden_") or key.startswith("liability_"):
            # get related clause index if present
            m = re.search(r'_(\d+)', key)
            if m:
                idx = int(m.group(1))
                original = clauses[idx]
                if use_openai_for_rewrite:
                    rewrite = openai_rewrite(original, instruction="Rewrite to remove risky phrasing and include a typical limitation of liability clause or safer phrasing.")
                else:
                    rewrite = CLAUSE_TEMPLATES.get("liability_cap") if "liability" in key else "Please revise this clause to align with policy. Example: " + CLAUSE_TEMPLATES.get("confidentiality", "")
                redlines[idx] = {"original": original[:1500], "suggestion": rewrite}
        else:
            # missing required clause -> suggest template
            if key in CLAUSE_TEMPLATES:
                redlines[f"add_{key}"] = {"suggestion": CLAUSE_TEMPLATES[key]}
            else:
                redlines[key] = {"suggestion": val}

    return {
        "clauses": clauses,
        "classification": classification,
        "policy_report": policy_report,
        "redlines": redlines
    }

# 11) Simple UI: upload files and run review
def run_colab_review(contract_value: float = None, use_openai_for_rewrite: bool = False):
    print("Upload contract files (PDF, DOCX, TXT, or images). You can upload multiple files.")
    uploaded = files.upload()
    if not uploaded:
        print("No files uploaded.")
        return
    full_text = ingest_uploaded_file(uploaded)
    print("\n--- Contract text length:", len(full_text), "characters ---\n")
    results = review_contract_text(full_text, contract_value=contract_value, use_openai_for_rewrite=use_openai_for_rewrite)

    # DataFrame summary of clauses, types, scores
    df_rows = []
    for i, clause in enumerate(results["clauses"]):
        ctype, score = results["classification"][i]
        df_rows.append({
            "clause_index": i,
            "predicted_type": ctype,
            "score": score,
            "snippet": clause[:300]
        })
    df = pd.DataFrame(df_rows)
    display(df)
    df.to_csv("summary.csv", index=False)
    #files.download("contract_review_summary.csv")

    print("\n--- Policy Report ---")
    pr = results["policy_report"]
    print("Found clause types:", pr["found"].keys())
    print("Missing required clauses:", pr["missing_required"])
    print("Forbidden phrase matches (index, phrase):", [(m[0], m[1]) for m in pr["forbidden_matches"]])
    print("Liability issues:", pr["liability_issues"])

    print("\n--- Suggested Redlines / Additions ---")
    for k, v in results["redlines"].items():
        print(f"\n>>> {k}")
        if "original" in v:
            print("Original (snippet):\n", v["original"][:500])
        print("Suggestion:\n", v["suggestion"][:1000])

    report = {
        "policy_report": pr,
        "redlines": results["redlines"]
    }
    import json
    with open("contract_review_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    #print("\nSaved report to contract_review_report.json")
    #files.download("contract_review_report.json")

    return df


#print("\nReady. To analyze a contract, call run_colab_review(contract_value=<number>, use_openai_for_rewrite=False)")

df = run_colab_review(1, use_openai_for_rewrite=False)

df.head()
