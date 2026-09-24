"""Frozen ASR inference and reference-free candidate evidence. No training operations."""
from .common import *
import copy,math
import numpy as np
import torch
import soundfile as sf
from peft import PeftModel
from transformers import WhisperForConditionalGeneration,WhisperProcessor,GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import LogitsProcessorList,SuppressTokensAtBeginLogitsProcessor,SuppressTokensLogitsProcessor
from .wer_rules_v3 import words
from rapidfuzz.distance import Levenshtein
from collections import Counter

class Engine:
 def __init__(self,seed,device='cuda:0'):
  self.device=device;self.seed=seed
  self.proc=WhisperProcessor.from_pretrained(str(MODEL),local_files_only=True)
  self.base=WhisperForConditionalGeneration.from_pretrained(str(MODEL),local_files_only=True,torch_dtype=torch.bfloat16)
  asset=ASSETS[f'F5-R0-{seed}'];self.model=PeftModel.from_pretrained(self.base,str(Path(asset['train_out_as_recorded'])/'adapter'),local_files_only=True).to(device).eval();self.model.requires_grad_(False)
  self.native=read(B6/'evidence/GENERATION_FREEZE.json')['native_generation_config']
  assert self.base.generation_config.to_dict()==self.native
  self.effective=read(B6/'runs/F6-TECH-TRAIN_dev_attempt0/EFFECTIVE_GENERATION.json')
  self.special=set(self.proc.tokenizer.all_special_ids)
 def features(self,path):
  x,sr=sf.read(path,dtype='float32')
  if x.ndim>1:x=x.mean(axis=1)
  if sr!=16000:
   from scipy.signal import resample_poly
   import math
   g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g).astype(np.float32)
  f=self.proc.feature_extractor([x],sampling_rate=16000,return_tensors='pt',padding='max_length',max_length=480000,truncation=True,return_attention_mask=True)
  return f.input_features.to(self.device,dtype=torch.bfloat16),f.attention_mask.to(self.device),dict(original_sample_rate=sr,original_duration_seconds=len(x)/16000,processed_frames=int(f.attention_mask.sum()))
 def prefix(self,lang):return [self.base.config.decoder_start_token_id]+[t for _,t in self.proc.get_decoder_prompt_ids(language=lang,task='transcribe')]
 def record(self,raw,rank,source):
  eos=self.base.config.eos_token_id;first=next((i for i,v in enumerate(raw) if v==eos),None)
  effective=raw[:first+1] if first is not None else raw
  body=effective[:-1] if first is not None else effective
  return dict(rank=rank,source=source,token_ids=body,raw_generated_tokens=len(effective),generated_tokens=len(body),cap=first is None and len(body)>=444,termination='natural_eos' if first is not None else 'length_cap' if len(body)>=444 else 'unknown',text=self.proc.tokenizer.decode(body,skip_special_tokens=True))
 @torch.inference_mode()
 def generate(self,feat,mask,lang,mode,encoder=None):
  if mode=='G':
   out=self.model.generate(input_features=feat,attention_mask=mask,language=lang,task='transcribe',max_new_tokens=444,do_sample=False,num_beams=1,num_return_sequences=1)
   ids=out[0].tolist();rec=self.record(ids,0,'G');rec['termination']='length_cap' if rec['cap'] else 'unknown_legacy_wrapper_removed_EOS';return [rec]
  gc=GenerationConfig.from_dict({k:v for k,v in self.effective[lang]['config'].items() if not k.startswith('_')})
  gc.num_beams=5;gc.num_return_sequences=5;gc.do_sample=False;gc.max_new_tokens=444
  processors=LogitsProcessorList([SuppressTokensAtBeginLogitsProcessor(self.native['begin_suppress_tokens'],begin_index=4,device=self.device),SuppressTokensLogitsProcessor(self.native['suppress_tokens'],device=self.device)])
  prefix=torch.tensor([self.prefix(lang)],device=self.device)
  kwargs=dict(attention_mask=mask,decoder_input_ids=prefix,generation_config=gc,logits_processor=processors)
  if encoder is None:kwargs['input_features']=feat
  else:kwargs['encoder_outputs']=encoder
  out=GenerationMixin.generate(self.base,**kwargs);assert out.shape[0]==5
  return [self.record(row[4:].tolist(),i,'B5') for i,row in enumerate(out)]
 @torch.inference_mode()
 def legacy_b5(self,feat,mask,lang):
  out=self.model.generate(input_features=feat,attention_mask=mask,language=lang,task='transcribe',max_new_tokens=444,do_sample=False,num_beams=5,num_return_sequences=1)
  return out[0].tolist()
 @torch.inference_mode()
 def encode(self,feat):return self.base.get_encoder()(feat,return_dict=True)
 @torch.inference_mode()
 def logps(self,feat,mask,lang,tokens,encoder=None):
  if not tokens:return [],[]
  prefix=self.prefix(lang);inp=torch.tensor([prefix+tokens[:-1]],device=self.device)
  kwargs=dict(decoder_input_ids=inp,attention_mask=mask,use_cache=False,return_dict=True)
  if encoder is None:kwargs['input_features']=feat
  else:kwargs['encoder_outputs']=encoder
  out=self.base(**kwargs);logits=out.logits[0,3:3+len(tokens)].float();assert logits.shape[0]==len(tokens)
  target=torch.tensor(tokens,device=self.device);values=torch.log_softmax(logits,dim=-1).gather(1,target[:,None]).squeeze(1).cpu().tolist()
  return values,[t not in self.special for t in tokens]

def perturb(feat,mask,uid,lang):
 n=int(mask.sum());lengths=[min(20,n-i) for i in range(0,n,20)];count=len(lengths)
 seed=int(hashlib.sha256(f'F7_AUDIO_BLOCKS_V1|{lang}|{uid}'.encode()).hexdigest()[:16],16)
 order=np.random.default_rng(seed).permutation(count).tolist()
 if count>=2 and order==list(range(count)):order=order[1:]+order[:1]
 out=feat.clone()
 if count>=2:out[:,:,:n]=torch.cat([feat[:,:,20*i:20*i+lengths[i]] for i in order],dim=-1)
 assert torch.equal(out[:,:,n:],feat[:,:,n:])
 return out,dict(valid_frames=n,block_lengths=lengths,order=order,nonidentity=count>=2,seed=seed)

from numba import njit
@njit(cache=True)
def _lcs_indices(a,b):
 n=len(a);m=len(b);dp=np.zeros((n+1,m+1),dtype=np.int16)
 for i in range(n-1,-1,-1):
  for j in range(m-1,-1,-1):dp[i,j]=dp[i+1,j+1]+1 if a[i]==b[j] else max(dp[i+1,j],dp[i,j+1])
 ma=np.zeros(n,dtype=np.bool_);mb=np.zeros(m,dtype=np.bool_);i=j=0
 while i<n and j<m:
  if a[i]==b[j]:ma[i]=True;mb[j]=True;i+=1;j+=1
  elif dp[i+1,j]>=dp[i,j+1]:i+=1
  else:j+=1
 return np.where(~ma)[0],np.where(~mb)[0]
def lcs_unmatched(a,b):
 aa,bb=_lcs_indices(np.asarray(a,dtype=np.int64),np.asarray(b,dtype=np.int64));return aa.tolist(),bb.tolist()

def deduplicate(beam,greedy,lang):
 result=[];mapping=[];seen={}
 for c in beam+greedy:
  key=tuple(words(c['text'],lang))
  if key not in seen:seen[key]=len(result);result.append(dict(c,candidate_index=len(result)))
  mapping.append(dict(source=c['source'],rank=c['rank'],unique_index=seen[key]))
 assert 1<=len(result)<=6
 return result,mapping

def assemble_features(candidates,lang):
 n=len(candidates)
 if n==1:return dict(features=[[0.]*10],probability=[1.],MBR=[0.],CD_seq=[0.],singleton=True)
 body=[];clean=[];audio=[];ww=[];base=[]
 for c in candidates:
  mask=c['body_mask'];tokens=[t for t,m in zip(c['token_ids'],mask) if m];cl=np.array([v for v,m in zip(c['clean_logp'],mask) if m]);au=np.array([x-y for x,y,m in zip(c['clean_logp'],c['perturbed_logp'],mask) if m]);w=words(c['text'],lang)
  body.append(tokens);clean.append(cl);audio.append(au);ww.append(w)
 sums=np.array([sum(v) for v in clean]);prob=np.exp(sums-sums.max());prob/=prob.sum()
 risk=[sum(prob[j]*Levenshtein.distance(ww[j],ww[k])/max(1,len(ww[j])) for j in range(n)) for k in range(n)]
 for i,c in enumerate(candidates):
  grams=[tuple(ww[i][j:j+4]) for j in range(max(0,len(ww[i])-3))];cnt=Counter(grams);rep=sum(cnt[g]>=3 for g in grams)/max(1,len(grams))
  base.append([float(clean[i].mean()) if len(clean[i]) else 0.,float(prob[i]),math.log1p(len(body[i])),math.log1p(len(ww[i])),rep,float(c['cap']),float(risk[i])])
 x=[];local_masks=[];global_audio=[float(v.mean()) if len(v) else 0. for v in audio]
 for k in range(n):
  ka,ba=lcs_unmatched(body[k],body[0]);m=lambda v,ix:float(v[ix].mean()) if ix else 0.
  x.append([v-z for v,z in zip(base[k],base[0])]+[m(clean[k],ka)-m(clean[0],ba),global_audio[k]-global_audio[0],m(audio[k],ka)-m(audio[0],ba)])
  local_masks.append(dict(candidate_unmatched_body_positions=ka,beam_unmatched_body_positions=ba))
 assert x[0]==[0.]*10 and np.isfinite(x).all()
 return dict(features=x,probability=prob.tolist(),MBR=risk,CD_seq=[base[i][0]+global_audio[i] for i in range(n)],singleton=False,local_masks=local_masks)
