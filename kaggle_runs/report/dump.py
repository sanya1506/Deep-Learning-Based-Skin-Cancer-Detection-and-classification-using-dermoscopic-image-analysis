import docx,sys
from docx.oxml.ns import qn
d=docx.Document('merged.docx')
body=d.element.body
out=open('dump_merged.txt','w',encoding='utf-8')
pi=ti=0
for el in body.iterchildren():
    if el.tag==qn('w:p'):
        p=docx.text.paragraph.Paragraph(el,d)
        t=p.text
        imgs=len(el.findall('.//'+qn('w:drawing')))
        out.write(f"P{pi} [{p.style.name}]{' IMG'*imgs} {t}\n"); pi+=1
    elif el.tag==qn('w:tbl'):
        tb=docx.table.Table(el,d)
        out.write(f"T{ti} rows={len(tb.rows)}\n")
        for r in tb.rows:
            out.write('   | '+' | '.join(c.text.replace('\n',' / ') for c in r.cells)+'\n')
        ti+=1
print(pi,'paragraphs',ti,'tables', len(d.inline_shapes),'inline images', len(d.sections),'sections')
