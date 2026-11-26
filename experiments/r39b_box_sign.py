"""R39b: does boxcar integration make attractive coupling (J < 0) harder to detect?
100 two-trap pairs (1/f, seed 471) generated twice, with and without 4x boxcar averaging (identical traps and
noise); paper statistic Lambda on each; power at the threshold matched to 5% of the independent partners,
split by the sign of J.
    python experiments/r39b_box_sign.py   -> results/r8/r39b_box_sign.json (+ _meta.json)
"""
import sys,json,os,numpy as np,torch
from multiprocessing import Pool
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1]); sys.path[:0]=[ROOT,ROOT+'/experiments']
os.environ.setdefault('RTN_MS','3'); sys.argv=['x','--r9']
import r8_calib as C
from rtnsim import sim2_torch as S
OUT=ROOT+'/results/r8/r39b_box_sign.json'
def make(p_box,n=100,seed=471):
    g=torch.Generator().manual_seed(seed); out=[]
    while len(out)<2*n:
        y,lab=S.make_batch(50,4096,g,"cpu",p_drive=0,p_box=p_box)
        if int(lab["K"][0])!=2: continue
        for r in range(y.shape[0]):
            if len(out)<2*n: out.append((y[r].double().numpy(),int(lab["coupled"][r]),float(lab["Jvec"][r].sum())))
    return out
def job(a):
    key,i,y=a
    try: s=C._r9_stat(y,2,np.random.default_rng(500+i))[0]
    except Exception as e: s=None
    return key,s
if __name__=='__main__':
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    meta={}; tasks=[]
    for pb in (1,0):
        recs=make(pb)                                   # same seed: same traps/noise, only integration differs
        for i,(y,c,J) in enumerate(recs):
            k=f'box{pb}:{i}'; meta[k]=dict(coupled=c,J=J)
            if k not in res: tasks.append((k,i,y))
    json.dump(meta,open(OUT.replace('.json','_meta.json'),'w'))
    print(len(tasks),flush=True)
    with Pool(2) as p:
        for n,(k,s) in enumerate(p.imap_unordered(job,tasks)):
            res[k]=s
            if n%20==0: json.dump(res,open(OUT,'w')); print(n,flush=True)
    json.dump(res,open(OUT,'w')); print('done',flush=True)
