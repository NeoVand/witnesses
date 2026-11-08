"""Exact requested operators. Axes: batch, token, score/value channel."""
import math
import jax
import jax.numpy as j


def pair_mask(n, causal=False, window=None, bos=False):
    ix=j.arange(n); i=ix[:,None,None]; a=ix[None,:,None]; b=ix[None,None,:]
    m=(a!=b)&(a!=i)&(b!=i)
    if causal:m=m&(a<i)&(b<i)
    if window is not None:m=m&(a>=i-window)&(b>=i-window)
    if bos:m=m&(a>0)&(b>0)
    return m


def probabilities(q,k1,k2,mask):
    s=j.einsum('bic,bjc,bkc->bijk',q,k1,k2)/math.sqrt(q.shape[-1])
    s=j.where(mask,s,-j.inf)
    valid=mask.any((-2,-1))
    s=j.where(valid[None,:,None,None],s,0)
    p=jax.nn.softmax(s.reshape(*s.shape[:2],-1),-1).reshape(s.shape)
    return j.where(mask,p,0)


def contract(p,v1,v2,full=False,sym=False):
    if sym:p=(p+p.swapaxes(-1,-2))/2
    if full:return j.einsum('bijk,bja,bkd->biad',p,v1,v2).reshape(*p.shape[:2],-1)
    return j.einsum('bijk,bja,bka->bia',p,v1,v2)


def moments(q,k1,k2,v1,v2,causal=False,bos=False,full=False):
    """Polarised power sums; query excluded, ordered distinct partners, exact Z."""
    n=q.shape[1]
    if bos:
        k1=k1.at[:,0].set(0);k2=k2.at[:,0].set(0)
    def others(t):
        return j.cumsum(t,axis=1)-t if causal else t.sum(1,keepdims=True)-t
    a=k1[...,None]*v1[:,:,None,:];b=k2[...,None]*v2[:,:,None,:]
    if full:
        diag=k1[:,:,:,None,None]*k2[:,:,:,None,None]*v1[:,:,None,:,None]*v2[:,:,None,None,:]
        pair=others(a)[...,None]*others(b)[:,:,:,None,:]-others(diag)
        num=j.einsum('bic,bicad->biad',q,pair).reshape(q.shape[0],n,-1)
    else:num=j.einsum('bic,bicv->biv',q,others(a)*others(b)-others(a*b))
    z=j.einsum('bic,bic->bi',q,others(k1)*others(k2)-others(k1*k2))
    counts=(j.arange(n)-(1 if bos else 0)) if causal else j.full((n,),n-1-(1 if bos else 0))
    if bos and not causal:counts=counts.at[0].add(1)
    valid=counts>=2
    safe=j.where(valid[None,:],z,1)
    return j.where(valid[None,:,None],num/safe[...,None],0),z


def feynman(q,u,w,v,L,R,mask,rounds=2):
    """branchg log-mass propagation, with constant leaf mass and oriented vertices.
    No tanh, per-round renormalization of N alone, or extra local gates.
    H=N/Z and F=log Z make this algebraically the stated N,Z recurrence.
    """
    n=q.shape[1]
    hopmask=mask.any(-1)
    # For n<3, allow legal unary edges even if there are no binary choices.
    if n<3:hopmask=~j.eye(n,dtype=bool)
    b=j.where(hopmask,j.einsum('bic,bjc->bij',q,u),-j.inf)
    c=j.where(mask,j.einsum('bic,bjc,bkc->bijk',q,u,w),-j.inf)
    H=v; F=j.zeros(v.shape[:2],v.dtype)
    for _ in range(rounds):
        unary=b+F[:,None,:]
        binary=c+F[:,None,:,None]+F[:,None,None,:]
        energies=j.concatenate([j.zeros((*F.shape,1),v.dtype),unary,binary.reshape(*F.shape,-1)],-1)
        F=jax.scipy.special.logsumexp(energies,-1)
        p=jax.nn.softmax(energies,-1)
        H=p[...,:1]*v+j.einsum('bij,bjv->biv',p[...,1:n+1],H)+contract(p[...,n+1:].reshape(q.shape[0],n,n,n),H@L,H@R)
    return H,F


def local_cubic(x,p,sym=False,window=16):
    """Exact local branch without constructing a sequence-length cubed tensor."""
    n=x.shape[1];ix=j.arange(n)[:,None]-j.arange(window,0,-1)[None,:]
    valid=ix>=0;ix=j.maximum(ix,0)
    q=x@p['q'];a=(x@p['k1'])[:,ix];b=(x@p['k2'])[:,ix]
    v1=(x@p['v1'])[:,ix];v2=(x@p['v2'])[:,ix]
    s=j.einsum('btc,btjc,btkc->btjk',q,a,b)/math.sqrt(q.shape[-1])
    m=valid[:,:,None]&valid[:,None,:]&(~j.eye(window,dtype=bool))[None]
    has=m.any((-1,-2));s=j.where(m,s,-j.inf);s=j.where(has[None,:,None,None],s,0)
    prob=jax.nn.softmax(s.reshape(*s.shape[:2],-1),-1).reshape(s.shape);prob=j.where(m,prob,0)
    if sym:prob=(prob+prob.swapaxes(-1,-2))/2
    y=j.einsum('btjk,btjv,btkv->btv',prob,v1,v2)
    return y@p['o']
