"""Convert the official FinGLM Chinese annotations into deterministic SFT JSONL."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data/finglm_cn'))
    ap.add_argument('--limit',type=int,default=1000); ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args(); raw=args.root/'raw'/'pre.jsonl'; rows=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines() if x.strip()]
    # Stable ID split: heldout questions are never used for SFT.
    rows=sorted(rows,key=lambda x:int(x['id']))
    heldout=[r for r in rows if int(r['id']) % 10 == 0]
    train=[r for r in rows if int(r['id']) % 10 != 0][:args.limit]
    def convert(r):
        answer=r.get('answer') or ['']
        return {'id':f"finglm_{int(r['id']):05d}", 'source_id':r['id'], 'question':r['question'],
                'answer':answer[0], 'type':r.get('type'), 'entity':r.get('prompt',{}).get('ent_name'),
                'year':r.get('prompt',{}).get('year'), 'dataset':'MetaGLM/FinGLM'}
    out=args.root/'sft'; out.mkdir(parents=True,exist_ok=True)
    train_rows=[convert(r) for r in train]; dev_rows=[convert(r) for r in heldout]
    for name,data in [('train',train_rows),('dev',dev_rows)]:
        with (out/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest={'dataset':'MetaGLM/FinGLM','upstream_revision':'ff48a9dc5e6c7f9d9a9c6120024dccefd8cbaa80','source_file_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(), 'train_count':len(train_rows),'dev_count':len(dev_rows),'limit':args.limit,'split_rule':'pre.jsonl id % 10 == 0 held out'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__': main()
