import json, pandas as pd, sys
from pathlib import Path

src = Path(sys.argv[1])            # applicants.json
dst = Path(sys.argv[2])            # applicants.csv

with open(src) as fh:
    raw = json.load(fh)

records = []
for app_id, sections in raw.items():
    cv_pt = sections.get("cv_pt", "") or sections.get("cv", "")
    cv_en = sections.get("cv_en", "")
    record = {
        "applicant_id": app_id,
        "cv_text": f"{cv_pt} {cv_en}".strip(),
        # se tiver texto da vaga, coloque aqui:
        "job_text": "",
    }
    records.append(record)

pd.DataFrame(records).to_csv(dst, index=False)
print(f"✅  CSV salvo em {dst}  ({len(records)} linhas)")