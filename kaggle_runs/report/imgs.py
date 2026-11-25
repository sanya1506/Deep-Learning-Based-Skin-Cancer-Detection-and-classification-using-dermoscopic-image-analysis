import docx
from docx.oxml.ns import qn
d=docx.Document('original.docx')
rels=d.part.rels
pi=0
for el in d.element.body.iterchildren():
    if el.tag==qn('w:p'):
        for bl in el.iter(qn('a:blip')):
            rid=bl.get(qn('r:embed')); ext=el.find('.//'+qn('wp:extent'))
            print(f"P{pi} {rid} {rels[rid].target_ref} {int(ext.get('cx'))/914400:.2f}x{int(ext.get('cy'))/914400:.2f}in")
        pi+=1
    elif el.tag==qn('w:tbl'):
        for bl in el.iter(qn('a:blip')): print('in table', bl.get(qn('r:embed')))
