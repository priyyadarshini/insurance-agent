"""Autonomous Insurance Claims Processing Agent

Lightweight rule-based FNOL processor for PDF/TXT documents.

What it does:
- Extracts key fields from FNOL documents
- Detects missing fields
- Classifies the claim routing
- Prints a JSON summary

Usage:
    python autonomous_insurance_claims_agent.py --input sample_fnol.txt
    python autonomous_insurance_claims_agent.py --input ./docs

Notes:
- Works best with text-based PDFs and clean TXT files.
- For scanned PDFs, OCR would be needed (not included here to keep it lightweight).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MANDATORY_FIELDS = [
    "Policy Number",
    "Policyholder Name",
    "Effective Dates",
    "Date",
    "Time",
    "Location",
    "Description",
    "Claimant",
    "Third Parties",
    "Contact Details",
    "Asset Type",
    "Asset ID",
    "Estimated Damage",
    "Claim Type",
    "Attachments",
    "Initial Estimate",
]

FRAUD_KEYWORDS = ["fraud", "inconsistent", "staged"]


# -----------------------------
# Text extraction helpers
# -----------------------------

def extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is not installed. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return extract_text_from_txt(path)
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


# -----------------------------
# Parsing helpers
# -----------------------------

def _first_match(patterns: List[str], text: str, flags=re.IGNORECASE | re.MULTILINE) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = match.group(1).strip()
            return re.sub(r"\s+", " ", value)
    return None


def extract_policy_number(text: str) -> Optional[str]:
    return _first_match([
        r"Policy\s*Number\s*[:\-]\s*([A-Z0-9\-\/]+)",
        r"Policy\s*No\.?\s*[:\-]\s*([A-Z0-9\-\/]+)",
        r"Policy\s*#\s*[:\-]\s*([A-Z0-9\-\/]+)",
    ], text)


def extract_policyholder_name(text: str) -> Optional[str]:
    return _first_match([
        r"Policyholder\s*Name\s*[:\-]\s*([A-Za-z][A-Za-z ,.'-]+)",
        r"Insured\s*Name\s*[:\-]\s*([A-Za-z][A-Za-z ,.'-]+)",
        r"Name\s*of\s*Policyholder\s*[:\-]\s*([A-Za-z][A-Za-z ,.'-]+)",
    ], text)


def extract_effective_dates(text: str) -> Optional[str]:
    return _first_match([
        r"Effective\s*Dates?\s*[:\-]\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4}\s*(?:to|\-|–)\s*[0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})",
        r"Coverage\s*Period\s*[:\-]\s*(.+)",
    ], text)


def extract_date(text: str) -> Optional[str]:
    return _first_match([
        r"Date\s*[:\-]\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})",
        r"Incident\s*Date\s*[:\-]\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})",
    ], text)


def extract_time(text: str) -> Optional[str]:
    return _first_match([
        r"Time\s*[:\-]\s*([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM|am|pm)?)",
        r"Incident\s*Time\s*[:\-]\s*([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM|am|pm)?)",
    ], text)


def extract_location(text: str) -> Optional[str]:
    return _first_match([
        r"Location\s*[:\-]\s*(.+)",
        r"Incident\s*Location\s*[:\-]\s*(.+)",
    ], text)


def extract_description(text: str) -> Optional[str]:
    return _first_match([
        r"Description\s*[:\-]\s*(.+)",
        r"Incident\s*Description\s*[:\-]\s*(.+)",
        r"Loss\s*Description\s*[:\-]\s*(.+)",
    ], text)


def extract_claimant(text: str) -> Optional[str]:
    return _first_match([
        r"Claimant\s*[:\-]\s*([A-Za-z][A-Za-z ,.'-]+)",
        r"Claimant\s*Name\s*[:\-]\s*([A-Za-z][A-Za-z ,.'-]+)",
    ], text)


def extract_third_parties(text: str) -> Optional[str]:
    return _first_match([
        r"Third\s*Part(?:y|ies)\s*[:\-]\s*(.+)",
        r"Involved\s*Third\s*Part(?:y|ies)\s*[:\-]\s*(.+)",
    ], text)


def extract_contact_details(text: str) -> Optional[str]:
    phone = _first_match([
        r"(?:Phone|Mobile|Contact)\s*[:\-]\s*([+()0-9\-\s]{7,})",
        r"(?:Phone|Mobile|Contact\s*Number)\s*[:\-]\s*([+()0-9\-\s]{7,})",
    ], text)
    email = _first_match([
        r"(?:Email|E-mail)\s*[:\-]\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    ], text)
    parts = []
    if phone:
        parts.append(f"Phone: {phone}")
    if email:
        parts.append(f"Email: {email}")
    return "; ".join(parts) if parts else None


def extract_asset_type(text: str) -> Optional[str]:
    return _first_match([
        r"Asset\s*Type\s*[:\-]\s*([A-Za-z][A-Za-z0-9 ,.'-]+)",
        r"Vehicle\s*Type\s*[:\-]\s*([A-Za-z][A-Za-z0-9 ,.'-]+)",
    ], text)


def extract_asset_id(text: str) -> Optional[str]:
    return _first_match([
        r"Asset\s*ID\s*[:\-]\s*([A-Za-z0-9\-\/]+)",
        r"Vehicle\s*ID\s*[:\-]\s*([A-Za-z0-9\-\/]+)",
        r"Property\s*ID\s*[:\-]\s*([A-Za-z0-9\-\/]+)",
    ], text)


def extract_estimated_damage(text: str) -> Optional[str]:
    return _first_match([
        r"Estimated\s*Damage\s*[:\-]\s*[₹Rs\.\s]*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        r"Damage\s*Estimate\s*[:\-]\s*[₹Rs\.\s]*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        r"Estimated\s*Loss\s*[:\-]\s*[₹Rs\.\s]*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    ], text)


def extract_claim_type(text: str) -> Optional[str]:
    return _first_match([
        r"Claim\s*Type\s*[:\-]\s*([A-Za-z][A-Za-z0-9 ,.'-]+)",
        r"Type\s*of\s*Claim\s*[:\-]\s*([A-Za-z][A-Za-z0-9 ,.'-]+)",
    ], text)


def extract_attachments(text: str) -> Optional[str]:
    return _first_match([
        r"Attachments?\s*[:\-]\s*(.+)",
        r"Supporting\s*Documents?\s*[:\-]\s*(.+)",
    ], text)


def extract_initial_estimate(text: str) -> Optional[str]:
    return _first_match([
        r"Initial\s*Estimate\s*[:\-]\s*[₹Rs\.\s]*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        r"First\s*Estimate\s*[:\-]\s*[₹Rs\.\s]*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    ], text)


def build_extracted_fields(text: str) -> Dict[str, Optional[str]]:
    return {
        "Policy Number": extract_policy_number(text),
        "Policyholder Name": extract_policyholder_name(text),
        "Effective Dates": extract_effective_dates(text),
        "Date": extract_date(text),
        "Time": extract_time(text),
        "Location": extract_location(text),
        "Description": extract_description(text),
        "Claimant": extract_claimant(text),
        "Third Parties": extract_third_parties(text),
        "Contact Details": extract_contact_details(text),
        "Asset Type": extract_asset_type(text),
        "Asset ID": extract_asset_id(text),
        "Estimated Damage": extract_estimated_damage(text),
        "Claim Type": extract_claim_type(text),
        "Attachments": extract_attachments(text),
        "Initial Estimate": extract_initial_estimate(text),
    }


# -----------------------------
# Routing logic
# -----------------------------

def is_missing(value: Optional[str]) -> bool:
    return value is None or str(value).strip() == ""


def to_number(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[,\s]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None


def route_claim(extracted: Dict[str, Optional[str]]) -> Tuple[str, str, List[str]]:
    missing = [field for field in MANDATORY_FIELDS if is_missing(extracted.get(field))]

    description = (extracted.get("Description") or "").lower()
    claim_type = (extracted.get("Claim Type") or "").lower()
    damage = to_number(extracted.get("Estimated Damage"))

    # Highest priority: missing mandatory fields
    if missing:
        return (
            "Manual review",
            "One or more mandatory fields are missing, so the claim needs manual review.",
            missing,
        )

    # Investigation keyword flag
    if any(keyword in description for keyword in FRAUD_KEYWORDS):
        return (
            "Investigation Flag",
            "The incident description contains a suspicious keyword such as fraud, inconsistent, or staged.",
            missing,
        )

    # Specialist queue for injury claims
    if claim_type == "injury" or "injury" in claim_type:
        return (
            "Specialist Queue",
            "The claim type is injury, so it should be routed to a specialist queue.",
            missing,
        )

    # Fast-track for small damages
    if damage is not None and damage < 25000:
        return (
            "Fast-track",
            "The estimated damage is below 25,000, so the claim qualifies for fast-track handling.",
            missing,
        )

    # Default route
    return (
        "Standard workflow",
        "No manual review condition was triggered, so the claim can proceed through the standard workflow.",
        missing,
    )


def process_document(path: Path) -> Dict[str, object]:
    text = load_document_text(path)
    extracted = build_extracted_fields(text)
    route, reasoning, missing = route_claim(extracted)

    return {
        "fileName": path.name,
        "extractedFields": extracted,
        "missingFields": missing,
        "recommendedRoute": route,
        "reasoning": reasoning,
    }


# -----------------------------
# CLI
# -----------------------------

def iter_documents(input_path: Path):
    if input_path.is_file():
        yield input_path
        return

    for p in sorted(input_path.iterdir()):
        if p.suffix.lower() in {".txt", ".pdf"} and p.is_file():
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Insurance Claims Processing Agent")
    parser.add_argument("--input", required=True, help="Path to a TXT/PDF file or a folder")
    parser.add_argument("--output", default="output.json", help="Output JSON file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    results = []
    for doc in iter_documents(input_path):
        try:
            results.append(process_document(doc))
        except Exception as e:
            results.append({
                "fileName": doc.name,
                "error": str(e),
            })

    output_data = results[0] if input_path.is_file() else {"documents": results}

    Path(args.output).write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
