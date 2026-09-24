"""Frozen singleton-mask PLL with complete centered windows and target-only MLM head."""
from .ir_common import *
import numpy as np
import copy
class PLL:
 def __init__(self,model_path,batch=32,device='cuda:0'):
  import torch
  from transformers import AutoTokenizer,AutoModelForMaskedLM
  self.torch=torch;self.batch=batch;self.device=device;self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
  self.model=AutoModelForMaskedLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to(device).eval();self.model.requires_grad_(False)
  # Per-position head has no cross-position operation. FP32 avoids reshaping-dependent bf16 head quantization.
  self.model.lm_head=copy.deepcopy(self.model.lm_head).float();self.special=set(self.tokenizer.all_special_ids)
  assert self.model.config.max_position_embeddings==514 and self.tokenizer.num_special_tokens_to_add(False)==2
 def tokenize(self,text):return [v for v in self.tokenizer.encode(text,add_special_tokens=False) if v not in self.special]
 def window(self,ids,t):
  n=len(ids);start=max(0,min(t-255,n-510));body=ids[start:start+510];pos=t-start+1
  seq=self.tokenizer.build_inputs_with_special_tokens(body);target=seq[pos];seq[pos]=self.tokenizer.mask_token_id
  assert target==ids[t] and 0<pos<len(seq)-1 and len(seq)<=512
  return seq,pos,target,start
 def _batch(self,jobs,verify_full=False):
  torch=self.torch;length=max(len(j[0]) for j in jobs);xx=torch.full((len(jobs),length),self.tokenizer.pad_token_id,dtype=torch.long,device=self.device);mask=torch.zeros_like(xx)
  for i,(seq,_,_,_) in enumerate(jobs):xx[i,:len(seq)]=torch.tensor(seq,device=self.device);mask[i,:len(seq)]=1
  pos=torch.tensor([j[1] for j in jobs],device=self.device);targets=torch.tensor([j[2] for j in jobs],device=self.device)
  with torch.inference_mode():
   hidden=self.model.roberta(input_ids=xx,attention_mask=mask,return_dict=True).last_hidden_state
   logits=self.model.lm_head(hidden[torch.arange(len(jobs),device=self.device),pos].float())
   vals=logits.float().log_softmax(-1).gather(1,targets[:,None]).squeeze(1)
   if verify_full:
    full=self.model.lm_head(hidden.float())[torch.arange(len(jobs),device=self.device),pos].log_softmax(-1).gather(1,targets[:,None]).squeeze(1)
    err=float((full-vals).abs().max());assert err<2e-4,err
    self.full_forward_check=dict(max_abs_logp_difference=err,all_positions_head_vs_masked_positions_head=True,tolerance=2e-4)
  return vals.cpu().double().tolist()
 def score_ids(self,ids,verify=False):
  if not ids:return dict(pll=0.,ntokens=0,windowed=False,target_count=0)
  vals=[]
  for start in range(0,len(ids),self.batch):
   jobs=[self.window(ids,t) for t in range(start,min(start+self.batch,len(ids)))];vals.extend(self._batch(jobs,verify_full=verify and start==0))
  assert len(vals)==len(ids) and np.isfinite(vals).all()
  return dict(pll=float(np.mean(vals,dtype=np.float64)),ntokens=len(ids),windowed=len(ids)>510,target_count=len(vals))
 def score_text(self,text):return self.score_ids(self.tokenize(text))
