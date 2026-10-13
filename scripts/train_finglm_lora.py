"""Small-memory QLoRA SFT for Chinese FinGLM question-answer pairs."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

def messages(question, answer=None):
    user={'role':'user','content':[{'type':'text','text':'你是中文企业年报问答助手。只能根据给出的年报证据回答；证据不足时必须只回答 UNANSWERABLE。涉及数值时保留原单位和精度，并在事实后标注证据页。问题：'+question}]}
    if answer is None: return [user]
    return [user, {'role':'assistant','content':[{'type':'text','text':answer}]}]

class SFT(Dataset):
    def __init__(self, path, processor, max_len):
        self.rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
        self.tok=processor.tokenizer; self.max_len=max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; question=r['question']
        if r.get('context'):
            question += '\n\n年报证据：\n' + r['context']
        full=self.tok.apply_chat_template(messages(question,r['answer']),tokenize=False,add_generation_prompt=False)
        prompt=self.tok.apply_chat_template(messages(question),tokenize=False,add_generation_prompt=True)
        full_ids=self.tok(full,add_special_tokens=False)['input_ids']
        prompt_ids=self.tok(prompt,add_special_tokens=False)['input_ids']
        answer_len=max(1, len(full_ids)-len(prompt_ids))
        if answer_len >= self.max_len:
            # Preserve a non-empty supervised suffix even for very long answers.
            full_ids=full_ids[-self.max_len:]; prompt_len=0
        else:
            prompt_len=min(len(prompt_ids), self.max_len-answer_len)
            full_ids=prompt_ids[-prompt_len:]+full_ids[-answer_len:]
        labels=[-100]*prompt_len+full_ids[prompt_len:]
        if not any(x != -100 for x in labels):
            raise ValueError(f"sample {r.get('id')} has no supervised answer tokens")
        return {'input_ids':full_ids,'attention_mask':[1]*len(full_ids),'labels':labels}

def collate(batch,pad):
    n=max(len(x['input_ids']) for x in batch); out={}
    for k in ['input_ids','attention_mask','labels']:
        fill=pad if k=='input_ids' else (0 if k=='attention_mask' else -100)
        out[k]=torch.tensor([x[k]+[fill]*(n-len(x[k])) for x in batch],dtype=torch.long)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/finglm_cn/sft/train.jsonl'); ap.add_argument('--output',default='models/adapters/finglm-qwen2vl-2b'); ap.add_argument('--model',default='Qwen/Qwen2-VL-2B-Instruct'); ap.add_argument('--init-adapter',default=None); ap.add_argument('--epochs',type=int,default=1); ap.add_argument('--batch-size',type=int,default=1); ap.add_argument('--grad-accum',type=int,default=8); ap.add_argument('--max-length',type=int,default=512); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for the 4-bit training profile')
    proc=AutoProcessor.from_pretrained(args.model,min_pixels=256*28*28,max_pixels=512*28*28)
    ds=SFT(args.data,proc,args.max_length)
    if args.limit: ds.rows=ds.rows[:args.limit]
    if args.smoke: ds.rows=ds.rows[:2]; args.epochs=1
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,collate_fn=lambda b:collate(b,proc.tokenizer.pad_token_id))
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=Qwen2VLForConditionalGeneration.from_pretrained(args.model,quantization_config=q,device_map={'':'cuda'},low_cpu_mem_usage=True)
    model=prepare_model_for_kbit_training(model); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj'])
    if args.init_adapter:
        model=PeftModel.from_pretrained(model,args.init_adapter,local_files_only=True,is_trainable=True)
    else:
        model=get_peft_model(model,lora)
    model.print_trainable_parameters(); model.train()
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
    step=0; opt.zero_grad(set_to_none=True); start=time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            batch={k:v.to('cuda') for k,v in batch.items()}; loss=model(**batch).loss/args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}; refusing to save an invalid adapter")
            loss.backward()
            if (step+1)%args.grad_accum==0 or step+1==len(loader): opt.step(); opt.zero_grad(set_to_none=True)
            step+=1
            if step%10==0: print(f'epoch={epoch+1} step={step}/{len(loader)} loss={loss.item()*args.grad_accum:.4f}',flush=True)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); proc.save_pretrained(out)
    (out/'training_summary.json').write_text(json.dumps({'data':str(args.data),'examples':len(ds),'epochs':args.epochs,'steps':step,'max_length':args.max_length,'elapsed_seconds':round(time.time()-start,1),'base_model':args.model,'init_adapter':args.init_adapter,'evidence_training':'context' in ds.rows[0] if ds.rows else False},ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',out)
if __name__=='__main__': main()
