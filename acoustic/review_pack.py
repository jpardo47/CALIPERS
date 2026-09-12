"""Create manual annotation material and a reader-only blinded pilot package.

No answers, masks or expert status are fabricated. The key is kept separately.
"""
import argparse,csv,json,shutil,hashlib
from pathlib import Path
import numpy as np
from .cli import read,write
from .pipeline import detect


READER='''<!doctype html><meta charset="utf-8"><title>Lectura de ecografías</title>
<style>body{font:18px system-ui;background:#17212c;color:white;max-width:1000px;margin:24px auto}img{max-width:100%;max-height:65vh;display:block;margin:20px 0}button,input,select{font:inherit;padding:8px;margin:5px}small{display:block}</style>
<h1>Evaluación visual de ecografías</h1><p>Piloto. Valore artefactos y distorsión; no emita diagnóstico para atención clínica.</p>
<label>ID del lector <input id="reader"></label><small>Las respuestas permanecen en este navegador hasta su exportación.</small>
<h2 id="caseid"></h2><img id="scan"><label>¿Detecta manipulación? <select id="manip"><option value="">Sin respuesta</option><option>Sí</option><option>No</option><option>Incierto</option></select></label>
<label>Distorsión del margen (0–4) <input id="distortion" type="number" min="0" max="4"></label>
<label>Confianza (1–5) <input id="confidence" type="number" min="1" max="5"></label>
<button id="prev">Anterior</button><button id="next">Guardar y siguiente</button><button id="export">Exportar respuestas</button>
<script>const cases=CASES;let index=0;const answers={};const el=id=>document.getElementById(id);
function show(){const c=cases[index];el('scan').src='images/'+c+'.png';el('caseid').textContent=(index+1)+' / '+cases.length+' · '+c;const a=answers[c]||{};for(const k of ['manip','distortion','confidence'])el(k).value=a[k]||'';}
function save(){if(!el('reader').value.trim()){alert('Indique su ID de lector');return false}answers[cases[index]]={reader_id:el('reader').value,case_id:cases[index],manip:el('manip').value,distortion:el('distortion').value,confidence:el('confidence').value,timestamp:new Date().toISOString()};return true;}
el('next').onclick=()=>{if(save()){index=Math.min(cases.length-1,index+1);show()}};el('prev').onclick=()=>{if(save()){index=Math.max(0,index-1);show()}};
el('export').onclick=()=>{if(!save())return;const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([JSON.stringify(Object.values(answers),null,2)],{type:'application/json'}));link.download='respuestas.json';link.click();URL.revokeObjectURL(link.href)};show();</script>'''


def build(manifest,cache,outdir):
    out=Path(outdir)
    if out.exists():raise ValueError('New review package required')
    rows=json.loads(Path(manifest).read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    annotation=out/'annotation';annotation.mkdir()
    records=[]
    # Review originals from full BUS_BRA, across devices, not only curated images.
    ranked=[]
    for row in rows:
        image=read(row['image']);mask=detect(image)
        yellow=0
        if image.ndim==3:
            rgb=image.astype(float)
            yellow=int(((rgb[...,0]>130)&(rgb[...,1]>100)&(rgb[...,2]<0.6*np.minimum(rgb[...,0],rgb[...,1]))).sum())
        ranked.append((yellow,int(mask.sum()),row))
    devices=sorted({r['device'] for r in rows});chosen=[]
    for device in devices:
        group=sorted([x for x in ranked if x[2]['device']==device],key=lambda x:(x[0],x[1]),reverse=True)
        chosen += [x[2] for x in group[:4]]
        chosen += [x[2] for x in group[-2:]]
    seen=set()
    for row in chosen:
        if row['image'] in seen:continue
        seen.add(row['image']);case=f'A{len(records)+1:03d}'
        image=read(row['image']);write(annotation/(case+'.png'),image)
        # Proposals named explicitly; never named GT.
        write(annotation/(case+'_proposal.png'),detect(image).astype(np.uint8)*255)
        records.append(dict(case_id=case,image=row['image'],patient_id=row['patient_id'],device=row['device'],
            reviewer_1='',reviewer_2='',adjudicator='',artifact_mask_path='',tissue_mask_path='',
            uncertain_mask_path='',clean_reviewed='',review_status='pending'))
    with (annotation/'review_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (annotation/'INSTRUCCIONES.md').write_text('''# Revisión pendiente: no hay GT experto de trazos

Las imágenes Axxx son originales a resolución nativa. Las propuestas Axxx_proposal NO son ground truth. Incluyen candidatos amarillos/blancos y posibles falsos positivos; no usarlas como autorización de borrado.

Dos revisores independientes deben dibujar máscaras binarias de artefactos y tejido, marcar incertidumbre y revisar la ausencia de marcas. El adjudicador resuelve discrepancias. Registrar IDs de revisores, rutas a PNG de tamaño original, fecha y estado adjudicated. No rellenar campos pendientes con respuestas automáticas.

Incluir ramas finas, antialiasing, texto de diferentes tamaños, calipers de borde, líneas punteadas y tejido hiperecoico/espiculado como negativos difíciles. La selección por brillo es un criterio de muestreo, no diagnóstico ni GT de negativo.

Las máscaras de lesión originales BUS_BRA pueden orientar una revisión anatómica separada, pero no indican qué píxeles son artefactos ni certifican parénquima sano.
''',encoding='utf-8')
    cached=json.loads((Path(cache)/'manifest.json').read_text(encoding='utf-8'))
    # One stimulus per patient prevents the reader recognizing matched duplicate views.
    unique={}
    for r in cached:unique.setdefault(r['patient_id'],r)
    selected=list(unique.values())[:60]
    rng=np.random.default_rng(6100);rng.shuffle(selected)
    variants=['reference']*(len(selected)//2)+['inpaint']*(len(selected)-len(selected)//2)
    rng.shuffle(variants)
    key=[];reader=out/'reader_only';(reader/'images').mkdir(parents=True)
    for i,(r,variant) in enumerate(zip(selected,variants)):
        case=hashlib.sha256(f'6100:{i}'.encode()).hexdigest()[:12]
        write(reader/'images'/(case+'.png'),read(Path(r['cache'])/(variant+'.png')))
        key.append(dict(case_id=case,variant=variant,patient_id=r['patient_id'],original=r['image']))
    (reader/'index.html').write_text(READER.replace('CASES',json.dumps([r['case_id'] for r in key])),encoding='utf-8')
    (out/'PRIVATE_reader_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    source=Path(rows[0]['image']).parent.parent/'LICENSE.txt'
    if source.exists():
        shutil.copyfile(source,reader/'LICENSE_BUS_BRA.txt')
        shutil.copyfile(source,annotation/'LICENSE_BUS_BRA.txt')
    (out/'status.json').write_text(json.dumps({'annotation_cases':len(records),'reader_cases':len(key),
        'expert_annotations_received':0,'reader_responses_received':0,
        'scope':'reader pilot, legacy reference not certified; CAD grid 128px',
        'delivery':'Share only reader_only folder, never PRIVATE_reader_key.json'},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--cache',required=True);p.add_argument('--outdir',required=True)
    a=p.parse_args();build(a.manifest,a.cache,a.outdir)
