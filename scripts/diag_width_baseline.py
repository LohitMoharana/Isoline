import sys; sys.path.insert(0,"/home/claude/isoline")
import numpy as np
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import scripts.replicate_rf as R
from isoline import data
from isoline.config import AAMI_MAP, CLASS_IDX
from isoline.metrics import report
D="/mnt/user-data/uploads/Isoline/data/mitdb"; ds1,_=data.get_splits(D)
def feats(rec, dv, base):
    sig,r,sym=data.load_mitdb_record(D,rec); x=resample_poly(sig,5,12)
    if dv: x=np.floor(x/dv)*dv
    r2=np.round(np.asarray(r)*150/360).astype(int); rr=np.diff(r2).astype(float)
    F,Y=[],[]; hr,hw=[],[]
    for i in range(1,len(r2)-1):
        c=AAMI_MAP.get(sym[i]); R0,Rm,Rp=rr[i-1],(rr[i-2] if i>=2 else rr[i-1]),rr[i]
        av=np.mean(hr[-32:]) if hr else R0
        s,e=r2[i]-56,r2[i]+40; ok=s>=0 and e<=len(x)
        if ok:
            seg=x[s:e]; seg=seg-(seg.mean() if base=="mean" else seg[0])
            q=seg[56-15:56+16]; pk=np.argmax(np.abs(q)); amp=q[pk]
            w=[float((q*np.sign(amp)>=f*abs(amp)).sum()) for f in (0.2,0.4,0.6)]
            pw=np.array(hw[-32:]) if hw else np.array([w]); wn=list(np.array(w)/np.maximum(pw.mean(0),1e-6)); hw.append(w)
        hr.append(R0)
        if not ok or c is None or c=="Q": continue
        F.append([R0/av,av/R0,Rp/R0,wn[1],Rm/R0,wn[0],wn[2],w[1]]); Y.append(CLASS_IDX[c])
    return F,Y
def run(dv,base):
    X,Y,G=[],[],[]
    for rec in ds1:
        f,y=feats(rec,dv,base); X+=f; Y+=y; G+=[int(rec)]*len(y)
    X,Y,G=np.array(X,np.float32),np.array(Y),np.array(G); p=np.zeros_like(Y)
    for tr,te in GroupKFold(22).split(X,Y,G): p[te]=RandomForestClassifier(40,random_state=0,n_jobs=-1).fit(X[tr],Y[tr]).predict(X[te])
    r=report(Y,p); print(f"dV {dv} baseline {base:5s} acc {100*r['accuracy']:.2f} S F1 {r['S']['f1']:.3f} V F1 {r['V']['f1']:.3f}",flush=True)
run(0,"mean"); run(0.1,"mean"); run(0,"start"); run(0.1,"start")
