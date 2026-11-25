"""Stream explicitly mapped CSV pairs without inventing labels or dropping relationships."""
import csv
import hashlib
from pathlib import Path


def read_pairs(path, resume_column, job_column, label_column=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        required = {resume_column, job_column} | ({label_column} if label_column else set())
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {sorted(required - set(reader.fieldnames or []))}')
        for row_number, row in enumerate(reader, 2):
            resume, job = row[resume_column], row[job_column]
            if not resume or not job:
                continue
            yield {'source_row': row_number, 'resume': resume, 'job': job,
                   'resume_id': hashlib.sha256(resume.encode('utf-8')).hexdigest(),
                   'job_id': hashlib.sha256(job.encode('utf-8')).hexdigest(),
                   'label': row[label_column] if label_column else None}
