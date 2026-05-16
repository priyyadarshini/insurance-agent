# Autonomous Insurance Claims Processing Agent

## Overview
This project is a lightweight FNOL (First Notice of Loss) processing agent that extracts insurance claim information from TXT/PDF documents and routes the claim to the appropriate workflow.

## Features
- Extracts key claim fields
- Detects missing mandatory fields
- Applies routing logic
- Generates structured JSON output

## Routing Rules
- Damage < 25000 → Fast-track
- Missing mandatory fields → Manual review
- Fraud-related keywords → Investigation Flag
- Injury claims → Specialist Queue

## Technologies Used
- Python
- Regex
- pypdf

## How to Run

Install dependency:

```bash
pip install pypdf
```text
To run a single file
```bash
py autonomous_insurance_claims_agent.py --input sample1.txt
```text
To run all files
```bash
py autonomous_insurance_claims_agent.py --input .
```text
Output

The system generates:
output.json
