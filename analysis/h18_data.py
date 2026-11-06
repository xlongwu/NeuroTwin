"""Data-scale conversion matching the evaluation loader's float32 raw path."""
import numpy as np


def calibrate_raw(raw: np.ndarray, length: int = 32) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(raw, dtype=np.float32).T)
    cal = x[:, :length]
    mu = cal.mean(axis=1, keepdims=True)
    scale = np.sqrt(cal.var(axis=1, keepdims=True) + 1e-6)
    return ((x-mu)/scale).T


def edges(series):
    x=np.asarray(series,dtype=np.float64);x=x-x.mean(axis=0)
    norm=np.sqrt((x*x).sum(0));den=norm[:,None]*norm[None,:]
    fc=np.full((116,116),np.nan);np.divide(x.T@x,den,out=fc,where=den>0)
    return fc[np.triu_indices(116,1)]

def similarity(left,right):
    left,right=np.stack(left),np.stack(right)
    if np.isfinite(left).all() and np.isfinite(right).all():
        a=left-left.mean(axis=1,keepdims=True);b=right-right.mean(axis=1,keepdims=True)
        denom=np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:]
        result=np.full(denom.shape,np.nan)
        np.divide(a@b.T,denom,out=result,where=denom>0)
        return result
    result=np.full((len(left),len(right)),np.nan)
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            good=np.isfinite(a)&np.isfinite(b)
            if good.sum()<3:continue
            x,y=a[good]-a[good].mean(),b[good]-b[good].mean()
            den=np.linalg.norm(x)*np.linalg.norm(y)
            if den>0:result[i,j]=x@y/den
    return result
