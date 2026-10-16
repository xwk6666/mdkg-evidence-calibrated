"""Standard-library metrics on frozen records; no model or inference imports."""
import hashlib
import math
from pathlib import Path


def require(condition,message):
    if not condition:raise ValueError(message)


def unique_index(rows,fields):
    fields=(fields,) if isinstance(fields,str) else tuple(fields)
    out={}
    for r in rows:
        key=tuple(str(r.get(k,'')) for k in fields)
        require(all(key) and key not in out,'Blank/duplicate stable key: '+','.join(fields))
        out[key]=r
    return out


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def auc(y,s):
    require(len(y)==len(s) and all(math.isfinite(float(x)) for x in s),'Invalid score array')
    p=sum(y);n=len(y)-p
    if not p or not n:return None
    order=sorted(zip(s,y));i=0;rank_sum=0.
    while i<len(order):
        j=i+1
        while j<len(order) and order[j][0]==order[i][0]:j+=1
        rank_sum+=(i+1+j)/2*sum(v for _,v in order[i:j]);i=j
    return (rank_sum-p*(p+1)/2)/(p*n)


def ap(y,s):
    positive=sum(y)
    if not positive:return None
    order=sorted(zip(s,y),reverse=True);i=tp=0;result=0.
    while i<len(order):
        j=i+1
        while j<len(order) and order[j][0]==order[i][0]:j+=1
        newly_positive=sum(v for _,v in order[i:j]);tp+=newly_positive
        result+=newly_positive/positive*tp/j;i=j
    return result


def legacy_constructed_ap(y,s):
    """Historical constructed-pair convention: stable descending rank, not grouped ties."""
    positive=sum(y)
    if not positive:return None
    order=sorted(zip(s,y),key=lambda item:item[0],reverse=True)
    hit=0;total=0.
    for rank,(_,label) in enumerate(order,1):
        if label:hit+=1;total+=hit/rank
    return total/positive


def prf(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def binary(y,s,pred):
    require(len(y)==len(s)==len(pred),'Different binary input lengths')
    tp=sum(a==b==1 for a,b in zip(y,pred));fp=sum(a==0 and b==1 for a,b in zip(y,pred))
    tn=sum(a==b==0 for a,b in zip(y,pred));fn=sum(a==1 and b==0 for a,b in zip(y,pred))
    specificity=tn/(tn+fp) if tn+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    denom=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return dict(n=len(y),positive_n=sum(y),negative_n=len(y)-sum(y),auroc=auc(y,s),ap=ap(y,s),
        tn=tn,specificity=specificity,balanced_accuracy=(specificity+recall)/2 if specificity is not None and recall is not None else None,
        mcc=(tp*tn-fp*fn)/math.sqrt(denom) if denom else 0.,
        negative_f1=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.,**prf(tp,fp,fn))


def calibration(y,q,bins=10):
    require(all(0<=v<=1 for v in q),'Probability outside [0,1]')
    ece=0.
    for b in range(bins):
        ids=[i for i,v in enumerate(q) if min(int(v*bins),bins-1)==b]
        if ids:ece+=len(ids)/len(q)*abs(sum(q[i] for i in ids)/len(ids)-sum(y[i] for i in ids)/len(ids))
    return dict(brier_score=sum((a-b)**2 for a,b in zip(y,q))/len(y),ece=ece,ece_bins=bins,ece_method='equal-width [0,1], final bin includes 1')


def quantile(values,q):
    values=sorted(values);p=(len(values)-1)*q;i=int(p);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(p-i)
