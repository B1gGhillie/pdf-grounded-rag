"""Select a small reproducible PDF subset matching FinGLM training questions."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data/finglm_cn')); ap.add_argument('--documents',type=int,default=25); args=ap.parse_args()
    rows=[json.loads(x) for x in (args.root/'raw'/'pre.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    rows=[r for r in rows if int(r['id'])%10 != 0]
    with (args.root/'raw'/'reports_list.csv').open(encoding='utf-8-sig') as f: reports=list(csv.DictReader(f))
    selected=[]; seen=set()
    for row in rows:
        p=row.get('prompt') or {}; entity=p.get('ent_name'); year=str(p.get('year',''))
        if not entity or (entity,year) in seen: continue
        hit=next((x for x in reports if entity in x['name'] and year in x['name']),None)
        if hit:
            selected.append({'doc_id':hit['name'][:-4].replace('/','_'),'source_name':hit['name'],'company':entity,'year':year,'upstream_sha256':None}); seen.add((entity,year))
        if len(selected)>=args.documents: break
    (args.root/'download_plan.json').write_text(json.dumps({'documents':selected},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'documents':len(selected),'plan':str(args.root/'download_plan.json')},ensure_ascii=False))
if __name__=='__main__': main()
