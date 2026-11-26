"""R39a: which fit-observable properties single out three-trap calibrated false positives?
Refits every three-trap record of the B = 99 calibration runs (200 independent: exp + exp_new; 100 coupled)
with the paper pipeline and stores step-to-noise, level spacing, step similarity, occupancy, dwell ratio and
AR(1) noise of the J = 0 fit.  Summary (Mann-Whitney, false positives vs other nulls) in main().
    python experiments/r39a_fp_features.py   -> results/r8/r39a_fp_features.json
"""
import sys,json,itertools,os,numpy as np
from multiprocessing import Pool
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1]); sys.path[:0]=[ROOT,ROOT+'/experiments']
os.environ.setdefault('RTN_MS','3'); sys.argv=['x','--r9']
import r8_calib as C, r8_oracle as O
from rtnsim.robustk import bits
OUT=ROOT+'/results/r8/r39a_fp_features.json'
def job(a):
    key,i,y,K=a
    rng=np.random.default_rng(500+i)
    try:
        s,f0,f1,beta,_=C._r9_stat(y,K,rng)
    except Exception as e:
        return key,dict(err=str(e))
    a_=np.abs(np.asarray(f0['coef'])[1:]); sig=float(f0['sig'])
    lv=np.sort(bits(K)@np.asarray(f0['coef'])[1:])
    c,e=np.asarray(f0['c']),np.asarray(f0['e']); occ=c/(c+e)
    path=np.asarray(f0['path']); occst=np.bincount(path,minlength=2**K)/len(path)
    return key,dict(stat=float(s),snr=float(a_.min()/sig),lev_sep=float(np.diff(lv).min()/sig),
        step_sim=float(min(abs(a_[p]-a_[q]) for p,q in itertools.combinations(range(K),2))/sig),
        occ_ext=float(np.max(np.abs(occ-0.5))),min_state_occ=float(occst.min()),
        tau_ratio=float(((c+e).max())/((c+e).min())),sig=sig,rho=float(f0.get('rho',0) or 0))
if __name__=='__main__':
    z=np.load(ROOT+'/results/r8/records.npz',allow_pickle=True); Y=z['y']
    ysN,_,_=O.make(100,seed=291)
    tasks=[]
    for tag,f,src in (('exp','r8_calib_exp_r9_ms3_B99.json',Y),('exp_new','r8_calib_exp_new_r9_ms3_B99.json',ysN),('coupled','r8_calib_coupled_r9_ms3_B99.json',Y)):
        d=json.load(open(ROOT+'/results/r8/'+f))
        for k,v in d.items():
            if isinstance(v,dict) and v.get('K')==3 and 'p_r8' in v:
                tasks.append((f'{tag}:{k}',int(k),src[int(k)],3))
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    tasks=[t for t in tasks if t[0] not in res]
    print(len(tasks),'to do',flush=True)
    with Pool(2) as p:
        for n,(k,r) in enumerate(p.imap_unordered(job,tasks)):
            res[k]=r
            if n%10==0: json.dump(res,open(OUT,'w')); print(n,flush=True)
    json.dump(res,open(OUT,'w')); print('done',flush=True)
