"""Balanced, role-specified tasks. No final examples are used for rate selection."""
import itertools
import numpy as np
TASKS=['typed','chronological','composition','who','match','set']
PERMS=list(itertools.permutations(range(3)))
MUL=np.array([[PERMS.index(tuple(a[b[i]] for i in range(3))) for b in PERMS] for a in PERMS])
G=np.array([[6*MUL[a//6,b//6]+MUL[a%6,b%6] for b in range(36)] for a in range(36)])
IDENTITY=0

def one(x,idx,val):x[np.arange(len(x)),idx,val]=1

def generate(task,seed,count):
    rng=np.random.default_rng(seed);xs=[];ys=[];need=[count//2,count-count//2]
    while min(need)>=0 and sum(need):
        B=max(256,sum(need)*2);x=np.zeros((B,10,64),np.float32);x[:,0,63]=1;x[:,9,62]=1
        if task in ('typed','chronological','match','set'):
            if task in ('typed','chronological'):
                ids=rng.integers(0,36,(B,9));roles=np.tile(np.array([1]*4+[2]*4+[0]),(B,1))
                if task=='typed':
                    for row in roles: rng.shuffle(row[:8])
                    for t in range(9):x[np.arange(B),t+1,36+roles[:,t]]=1
                else:
                    for t in range(9):x[:,t+1,40+t]=1
                for t in range(9):
                    x[np.arange(B),t+1,ids[:,t]//6]=1;x[np.arange(B),t+1,6+ids[:,t]%6]=1
                a=ids[:,:8];q=ids[:,8];rel=G[G[q[:,None,None],a[:,:,None]],a[:,None,:]]==IDENTITY
                if task=='typed':m=(roles[:,:8,None]==1)&(roles[:,None,:8]==2)
                else:m=np.triu(np.ones((8,8),bool),1)[None]
                y=(rel&m).any((1,2)).astype(int)
            elif task=='match':
                ids=rng.integers(0,31,(B,9))
                for t in range(9):x[np.arange(B),t+1,ids[:,t]]=1
                y=(((ids[:,8,None,None]+ids[:,:8,None]+ids[:,None,:8])%31==0)&(~np.eye(8,dtype=bool))[None]).any((1,2)).astype(int)
            else:
                ids=np.argsort(rng.random((B,81)),axis=1)[:,:9]
                attrs=np.stack([(ids//3**a)%3 for a in range(4)],-1)
                for t in range(9):
                    for a in range(4):x[np.arange(B),t+1,3*a+attrs[:,t,a]]=1
                y=((((attrs[:,8,None,None,:]+attrs[:,:8,None,:]+attrs[:,None,:8,:])%3==0).all(-1))&(~np.eye(8,dtype=bool))[None]).any((1,2)).astype(int)
        elif task=='composition':
            f=np.array([rng.permutation(4) for _ in range(B)]);g=np.array([rng.permutation(4) for _ in range(B)]);q=rng.integers(0,4,B)
            right=f[np.arange(B),g[np.arange(B),q]];reverse=g[np.arange(B),f[np.arange(B),q]]
            y=rng.integers(0,2,B);candidate=np.where(y,right,reverse)
            for t in range(8):
                x[:,t+1,t%4]=1;x[np.arange(B),t+1,4+(g if t<4 else f)[:,t%4]]=1;x[:,t+1,36+(t//4)]=1
            x[np.arange(B),9,8+q]=1;x[np.arange(B),9,12+candidate]=1
            y=np.where(right!=reverse,y,-1)
        elif task=='who':
            a=rng.integers(0,8,(B,4));b=(a+rng.integers(1,8,(B,4)))%8;verb=rng.integers(0,4,B);role=rng.integers(0,2,B)
            good=np.where(role,a[np.arange(B),verb],b[np.arange(B),verb]);bad=np.where(role,b[np.arange(B),verb],a[np.arange(B),verb])
            y=rng.integers(0,2,B);candidate=np.where(y,good,bad)
            for t in range(8):
                x[np.arange(B),t+1,(a if t%2==0 else b)[:,t//2]]=1;x[:,t+1,8+t//2]=1;x[:,t+1,36+t%2]=1
            x[np.arange(B),9,8+verb]=1;x[np.arange(B),9,16+candidate]=1;x[np.arange(B),9,38+role]=1
            # Shuffle complete role tokens; chronology is irrelevant here.
            for row in x:rng.shuffle(row[1:9])
        else:raise ValueError(task)
        for label in (0,1):
            ix=np.flatnonzero(y==label)[:need[label]];xs.append(x[ix]);ys.append(y[ix]);need[label]-=len(ix)
    x=np.concatenate(xs);y=np.concatenate(ys).astype(np.int32);order=rng.permutation(count)
    return x[order],y[order]
