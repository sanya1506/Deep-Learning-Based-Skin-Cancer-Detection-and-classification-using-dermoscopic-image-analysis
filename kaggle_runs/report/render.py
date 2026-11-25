import fitz,sys
doc=fitz.open(sys.argv[1]); out=sys.argv[2]
import os; os.makedirs(out,exist_ok=True)
pages=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else range(1,len(doc)+1)
for p in pages:
    doc[p-1].get_pixmap(dpi=70).save(f"{out}/p{p:02d}.png")
# find pages containing key strings
for key in ["Table 6.1:","Figure 6.1:","Table 6.10:","Table 6.13:","Table 5.4:"]:
    print(key,[i+1 for i in range(len(doc)) if key in doc[i].get_text()])
