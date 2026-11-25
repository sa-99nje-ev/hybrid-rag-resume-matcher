"""Convert the authors' released dataset to local document JSONL, retaining source IDs."""
import csv
import json
from pathlib import Path
from docx import Document

root=Path(__file__).resolve().parents[1]
source=root/'data/raw/vacancy-resume-matching-dataset'
out=root/'data/processed/documents.jsonl'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8',newline='\n') as target:
    for path in sorted((source/'CV').glob('*.docx'),key=lambda p:int(p.stem)):
        doc=Document(path)
        # DOCX paragraphs and tables are retained in document order.
        from docx.document import Document as DocType
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        body=doc.element.body
        texts=[]
        for child in body.iterchildren():
            if child.tag == qn('w:p'): texts.append(Paragraph(child,doc).text)
            elif child.tag == qn('w:tbl'):
                table=Table(child,doc)
                texts.extend('\t'.join(cell.text for cell in row.cells) for row in table.rows)
        target.write(json.dumps({'id':'cv:'+path.stem,'kind':'resume','text':'\n'.join(texts),'source':str(path.relative_to(root))},ensure_ascii=False)+'\n')
    with (source/'5_vacancies.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            target.write(json.dumps({'id':'job:'+row['id'],'kind':'job','text':row['job_description'],'source':'5_vacancies.csv','source_uid':row['uid'],'title':row['job_title']},ensure_ascii=False)+'\n')
print('Wrote local corpus to',out)
