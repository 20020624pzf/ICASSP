from pathlib import Path
import os,sys,json,hashlib,time
HERE=Path(__file__).resolve().parents[1]
CFG=json.loads(Path(os.environ["ASR_REPRO_CONFIG"]).read_text())
MODEL=Path(CFG["whisper_snapshot"])
B6=HERE/"bindings"
ASSETS={f"F5-R0-{seed}":{"train_out_as_recorded":str(Path(path).resolve().parent)} for seed,path in CFG["adapter_directories"].items()}
LANGS=["ar","fa","pl","tr","it"]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
