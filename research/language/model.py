"""Language-model backbone and the pair branches added to it.

A pre-LayerNorm GPT decoder (hoa_lm) with options: rotary positions on queries and keys (pos='rope'), RMSNorm,
SwiGLU, and a branch added after attention in every `branch_every`-th block (x <- x + Attn(LN x); x <- x + B(LN' x);
x <- x + MLP(LN x)). The join (arm 'cubic') scores the ordered pairs (j, k), j != k, of the `window` strictly previous
tokens by sum_c q_c k1_jc k2_kc / sqrt(r), takes a softmax over the pairs and returns sum_jk p_jk v1_j * v2_k.
Branch projections are bias-free and every branch starts with zero output, so each model starts as its baseline.
"""
import math
import jax
import jax.numpy as j
import numpy as np
import hoa_lm as lm
import cubics

ARMS = ['gpt','cubic','sym','tie_k','tie_v','tie_both','full',
        'shared_full','wedge','symmetric','newton','signed','signed_full',
        'pairwise','positive','feynman','feynman_tied','cubic_mass','feynman_r1','oriented_exp','feynman_s1','feynman_s2','cx','band','qband','band2','topm','pstream','local']


def config(arm='gpt', **kw):
    c = lm.build_parser().parse_args([])
    for k,v in {**dict(d=512,layers=8,heads=8,ctx=256,batch=64,vocab=50304,
                    bf16=1,mlp=2048,arch='gpt',window=16,branch_heads=4,
                    branch_every=4,arm=arm,norm='ln',pos='learned',mlp_kind='gelu',dp=64,m=16,offsets=(1,)),**kw}.items():
        setattr(c,k,v)
    return c

NEW=('cx','band','qband','band2','topm','pstream','local')

def dims(arm):
    if arm.startswith('feynman'): return 32,64,64
    if arm in ['full','signed_full']: return 32,18,324
    if arm in ['shared_full','wedge','symmetric']: return 12,20,400
    if arm in ['pairwise','newton']: return 114,114,114
    return 32,120,120

def init(c,seed):
    p=lm.init(jax.random.PRNGKey(seed),c)
    backbone(c,p,seed)
    H,D=c.branch_heads,c.d;r,v,out=dims(c.arm)
    if getattr(c,'local_every',0):
        for i,l in enumerate(p['layers']):
            if (i+1)%c.local_every==0:
                ks=iter(jax.random.split(jax.random.PRNGKey(seed+15485863+i),4))
                w=lambda a,b:jax.random.normal(next(ks),(a,b))/math.sqrt(a)
                l['local']=dict(norm=lm.ln_init(D),v1=w(D,H*120),v2=w(D,H*120),o=j.zeros((H*120,D)))
    if getattr(c,'mulatt',0):
        for i,l in enumerate(p['layers']):
            if (i+1)%c.mulatt==0: l['mg']=j.zeros((D,D))
    if c.arm=='gpt': return p
    for i,l in enumerate(p['layers']):
        if (i+1)%c.branch_every: continue
        keys=iter(jax.random.split(jax.random.PRNGKey(seed+7919+i),12))
        if c.arm in NEW:
            l['branch']=cubics.init(c,keys,D);continue
        weight=lambda a,b:jax.random.normal(next(keys),(a,b))/math.sqrt(a)
        b={'norm':lm.ln_init(D)}
        if c.arm=='newton':
            b.update(th=weight(D,H*r),g=weight(D,H*r),o=weight(H*2*r,D),alpha=j.zeros((4,D)))
        elif c.arm=='pairwise':
            b.update(q=weight(D,H*r),k=weight(D,H*r),v=weight(D,H*v),o=j.zeros((H*out,D)))
        else:
            b.update(q=weight(D,H*r),k1=weight(D,H*r),k2=weight(D,H*r),v1=weight(D,H*v))
            if c.arm.startswith('feynman'):
                b['L']=jax.random.normal(next(keys),(H,v,v))/math.sqrt(v)
                b['R']=jax.random.normal(next(keys),(H,v,v))/math.sqrt(v)
            elif c.arm not in ['shared_full','wedge','symmetric']: b['v2']=weight(D,H*v)
            b['o']=j.zeros((H*out,D))
            if c.arm=='cubic_mass': b['w']=j.zeros((H,D))
        l['branch']=b
    return p

def local_weights(q,k1,k2,window):
    # B,T,H,C -> B,T,H,J,K. Both partner axes have the same strict-past window.
    T=q.shape[1];ix=j.arange(T)[:,None]-j.arange(window,0,-1)[None,:]
    valid=ix>=0;ix=j.maximum(ix,0)
    a=k1[:,ix];b=k2[:,ix]
    s=j.einsum('bthc,btjhc,btkhc->bthjk',q,a,b)/math.sqrt(q.shape[-1])
    mask=valid[:,:,None]&valid[:,None,:]&(~j.eye(window,dtype=bool))[None]
    s=j.where(mask[None,:,None],s,-j.inf)
    s=j.where(mask.any((-1,-2))[None,:,None,None,None],s,0)
    p=jax.nn.softmax(s.reshape(*s.shape[:3],-1),-1).reshape(s.shape)
    return j.where(mask[None,:,None],p,0),ix

def local_mass(q,k1,k2,window):
    # Log mean exponentiated score per head: what softmax divides away. Zero with no valid pair.
    T=q.shape[1];ix=j.arange(T)[:,None]-j.arange(window,0,-1)[None,:]
    valid=ix>=0;ix=j.maximum(ix,0)
    s=j.einsum('bthc,btjhc,btkhc->bthjk',q,k1[:,ix],k2[:,ix])/math.sqrt(q.shape[-1])
    mask=valid[:,:,None]&valid[:,None,:]&(~j.eye(window,dtype=bool))[None]
    s=j.where(mask[None,:,None],s,-j.inf)
    count=mask.sum((-1,-2))
    safe=j.where((count>0)[None,:,None,None,None],s,0)
    m=jax.scipy.special.logsumexp(safe.reshape(*s.shape[:3],-1),-1)-j.log(j.maximum(count,1))[None,:,None]
    return j.where((count>0)[None,:,None],m,0)

def local_values(p,v1,v2,ix,kind='hadamard'):
    if kind=='sym': p=(p+p.swapaxes(-1,-2))/2
    a,b=v1[:,ix],v2[:,ix]
    if kind in ['hadamard','sym']:
        return j.einsum('bthjk,btjhv,btkhv->bthv',p,a,b)
    m=j.einsum('bthjk,btjhu,btkhv->bthuv',p,a,b)
    if kind=='wedge': m=(m-m.swapaxes(-1,-2))/2
    if kind=='symmetric': m=(m+m.swapaxes(-1,-2))/2
    return m.reshape(*m.shape[:3],-1)

def signed_moments(q,k1,k2,v1,v2,full=False,positive=False):
    # Pair moments by causal prefix sums, with a separate head axis.
    prefix=lambda a:j.cumsum(a,axis=1)-a
    a=k1[...,None]*v1[:,:,:,None,:]
    b=k2[...,None]*v2[:,:,:,None,:]
    if full:
        diag=(k1*k2)[...,None,None]*v1[:,:,:,None,:,None]*v2[:,:,:,None,None,:]
        pair=prefix(a)[...,None]*prefix(b)[:,:,:,:,None,:]-prefix(diag)
        y=j.einsum('bthc,bthcuv->bthuv',q,pair).reshape(*q.shape[:3],-1)
    else:
        y=j.einsum('bthc,bthcv->bthv',q,prefix(a)*prefix(b)-prefix(a*b))
    valid=(j.arange(q.shape[1])>=2)[None,:,None,None]
    if positive:
        z=j.einsum('bthc,bthc->bth',q,prefix(k1)*prefix(k2)-prefix(k1*k2))
        y=y/j.where(valid[...,0],j.maximum(z,1e-12),1)[...,None]
    return j.where(valid,y,0)

def old_newton(phi):
    # Causal power sums of orders 1..4 and per-order normalisation.
    P=[j.cumsum(phi**k,1)-phi**k for k in range(1,4)]
    e=[j.ones_like(phi)]
    for m in range(1,4):
        e.append(sum((-1)**(k-1)*e[m-k]*P[k-1] for k in range(1,m+1))/m)
    out=[]
    for m in range(4):
        div=j.asarray([1/max(math.comb(t,m),1) if t>=m else 0 for t in range(phi.shape[1])])[None,:,None,None]
        f=phi*e[m]*div;f=j.concatenate([f.real,f.imag],-1).reshape(*phi.shape[:2],-1)
        out.append(f/j.sqrt(j.mean(f*f,-1,keepdims=True)+1e-12))
    return out

def local_feynman(q,u,w,v,L,R,window,rounds=2,scale=1.):
    """Exact leaf/unary/binary ordered-tree recurrence in stable log-mass form."""
    B,T,H,_=q.shape;ix=j.arange(T)[:,None]-j.arange(window,0,-1)[None,:]
    valid=ix>=0;ix=j.maximum(ix,0)
    unary=j.einsum('bthc,btjhc->bthj',q,u[:,ix])*scale
    unary=j.where(valid[None,:,None],unary,-j.inf)
    binary=j.einsum('bthc,btjhc,btkhc->bthjk',q,u[:,ix],w[:,ix])*scale
    mask=valid[:,:,None]&valid[:,None,:]&(~j.eye(window,dtype=bool))[None]
    binary=j.where(mask[None,:,None],binary,-j.inf)
    h=v;f=j.zeros((B,T,H),v.dtype)
    for _ in range(rounds):
        past=f[:,ix].transpose(0,1,3,2)
        energies=j.concatenate([j.zeros((B,T,H,1),v.dtype),unary+past,
             (binary+past[...,None]+past[...,None,:]).reshape(B,T,H,-1)],-1)
        f=jax.scipy.special.logsumexp(energies,-1);p=jax.nn.softmax(energies,-1)
        a=j.einsum('bthv,hvu->bthu',h,L);b=j.einsum('bthv,hvu->bthu',h,R)
        h=(p[...,:1]*v+j.einsum('bthj,btjhv->bthv',p[...,1:window+1],h[:,ix])
           +local_values(p[...,window+1:].reshape(B,T,H,window,window),a,b,ix))
    return h,f

def branch(c,b,x,stats=None):
    if c.arm=='cx': return cubics.window_cubic(c,b,x,stats)
    if c.arm in ('band','qband','band2'): return cubics.band(c,b,x,stats)
    if c.arm=='topm': return cubics.topm(c,b,x,stats)
    if c.arm=='local': return cubics.local(c,b,x,stats)
    B,T,D=x.shape;H=c.branch_heads;r,v,_=dims(c.arm)
    x=norm(c,b['norm'],x)
    proj=lambda name,width:lm.mm(x,b[name]).reshape(B,T,H,width)
    if c.arm=='newton':
        phi=jax.nn.sigmoid(proj('g',r))*j.exp(2j*proj('th',r))
        return sum(lm.mm(f,b['o'])*b['alpha'][m] for m,f in enumerate(old_newton(phi)))
    q=proj('q',r)
    if c.arm.startswith('feynman'):
        y,_=local_feynman(q,proj('k1',r),proj('k2',r),proj('v1',v),b['L'],b['L'] if c.arm=='feynman_tied' else b['R'],c.window,1 if c.arm in ('feynman_r1','feynman_s1') else 2,1/math.sqrt(r) if c.arm in ('feynman_s1','feynman_s2') else 1.)
        return lm.mm(y.reshape(B,T,-1),b['o'])
    if c.arm=='pairwise':
        ix=j.arange(T)[:,None]-j.arange(c.window,0,-1)[None,:];valid=ix>=0;ix=j.maximum(ix,0)
        s=j.einsum('bthc,btjhc->bthj',q,proj('k',r)[:,ix])/math.sqrt(r)
        s=j.where(valid[None,:,None],s,-j.inf);s=j.where(valid.any(-1)[None,:,None,None],s,0)
        prob=j.where(valid[None,:,None],jax.nn.softmax(s,-1),0)
        y=j.einsum('bthj,btjhv->bthv',prob,proj('v',v)[:,ix])
    else:
        k1=proj('k1',r);k2=k1 if c.arm in ['tie_k','tie_both'] else proj('k2',r)
        v1=proj('v1',v);v2=v1 if c.arm in ['tie_v','tie_both','shared_full','wedge','symmetric'] else proj('v2',v)
        if c.arm in ['signed','signed_full','positive','oriented_exp']:
            if c.arm=='positive': q,k1,k2=map(jax.nn.softplus,[q,k1,k2])
            if c.arm=='oriented_exp': q,k1,k2=[j.exp(j.clip(t,-6.,6.)) for t in (q,k1,k2)]
            y=signed_moments(q,k1,k2,v1,v2,c.arm=='signed_full',c.arm in ['positive','oriented_exp'])
            if c.arm not in ['positive','oriented_exp']:
                flat=y.reshape(B,T,-1);flat=flat/j.sqrt(j.mean(flat*flat,-1,keepdims=True)+1e-6)
                y=flat
        else:
            prob,ix=local_weights(q,k1,k2,c.window)
            kind=c.arm if c.arm in ['sym','wedge','symmetric'] else 'full' if c.arm in ['full','shared_full'] else 'hadamard'
            y=local_values(prob,v1,v2,ix,kind)
            if stats is not None and kind=='hadamard':
                a,b2=v1[:,ix],v2[:,ix]
                fac=j.einsum('bthj,btjhv->bthv',prob.sum(-1),a)*j.einsum('bthk,btkhv->bthv',prob.sum(-2),b2)
                stats['factorised_share']=(fac**2).sum()/((y**2).sum()+1e-12)
                flat=prob[:,c.window:].reshape(B,T-c.window,H,-1)
                stats['effective_pairs']=j.exp(-(flat*j.log(flat+1e-12)).sum(-1)).mean()
                stats['top_pair']=flat.max(-1).mean()
            if c.arm=='cubic_mass':
                return lm.mm(y.reshape(B,T,-1),b['o'])+lm.mm(local_mass(q,k1,k2,c.window),b['w'])
    return lm.mm(y.reshape(B,T,-1),b['o'])


def rope(t,base=10000.):
    # t: B,H,T,C with even C. Rotates channel pairs by position times a geometric frequency.
    T,C=t.shape[2],t.shape[3]
    freq=base**(-j.arange(0,C,2)/C)
    angle=j.arange(T)[None,None,:,None]*freq[None,None,None,:]
    cos,sin=j.cos(angle),j.sin(angle)
    a,b=t[...,0::2],t[...,1::2]
    return j.stack([a*cos-b*sin,a*sin+b*cos],-1).reshape(t.shape)

def norm(c,p,x):
    if c.norm=='rms':
        return x/j.sqrt((x*x).mean(-1,keepdims=True)+1e-5)*p['g']
    return lm.ln(p,x)

def backbone(c,p,seed):
    """Backbone options on top of the hoa_lm parameters. Options off leave every array untouched."""
    keys=iter(jax.random.split(jax.random.PRNGKey(seed+104729),6*len(p['layers'])+4))
    if c.pos=='rope': p.pop('pos')
    if getattr(c,'ngram',0):
        n=c.d//2
        p['ng_a'],p['ng_b']=lm.dense(next(keys),c.d,n),lm.dense(next(keys),c.d,n)
        p['ng_o']=j.zeros((n,c.d))
    for l in p['layers']:
        if getattr(c,'mix',0):
            w=c.mix*c.d//c.heads
            l['mix_k2']=lm.dense(next(keys),c.d,w);l['mix_v2']=lm.dense(next(keys),c.d,w)
        if c.mlp_kind=='swiglu':
            h=c.mlp*2//3;D=c.d;L=c.layers
            l['fc1']=lm.dense(next(keys),D,h);l['gate']=lm.dense(next(keys),D,h)
            l['fc2']=lm.dense(next(keys),h,D,scale=1/math.sqrt(2*L))
    return p

def attention(c,l,x,stats=None):
    B,T,D=x.shape;H=c.heads
    q,k,v=lm.project_qkv(l,x)
    sh=lambda t:t.reshape(B,T,H,D//H).transpose(0,2,1,3)
    q,k,v=sh(q),sh(k),sh(v)
    if getattr(c,'mix',0):
        # The last `mix` heads use product keys and values with the previous token.
        n=c.mix;first=H-n
        sub=lambda t:t.reshape(B,T,n,D//H).transpose(0,2,1,3)
        prev=lambda t:j.where((j.arange(T)>=1)[None,None,:,None],j.roll(t,1,axis=2),0.)
        k2,v2=sub(lm.lin(l['mix_k2'],x)),sub(lm.lin(l['mix_v2'],x))
        k=k.at[:,first:].set(k[:,first:]*prev(k2));v=v.at[:,first:].set(v[:,first:]*prev(v2))
    if c.pos=='rope':
        rq,rk=rope(q),rope(k)
        if getattr(c,'mix_norot',0):
            first=H-c.mix
            rq,rk=rq.at[:,first:].set(q[:,first:]),rk.at[:,first:].set(k[:,first:])
        q,k=rq,rk
    s=lm.mm(q,k,'bhtc,bhsc->bhts')/math.sqrt(q.shape[-1])
    s=j.where(j.tril(j.ones((T,T),bool)),s,-1e9)
    A=jax.nn.softmax(s,-1)
    if stats is not None:
        tail=A[:,:,T//2:]
        stats['attn_partners']=j.exp(-(tail*j.log(tail+1e-12)).sum(-1)).mean()
        stats['attn_top']=tail.max(-1).mean()
    return lm.lin(l['o'],lm.mm(A,v,'bhts,bhsc->bhtc').transpose(0,2,1,3).reshape(B,T,D))

def mlp(c,l,x):
    if c.mlp_kind=='swiglu':
        return lm.lin(l['fc2'],jax.nn.silu(lm.lin(l['gate'],x))*lm.lin(l['fc1'],x))
    return lm.lin(l['fc2'],jax.nn.gelu(lm.lin(l['fc1'],x)))

def forward(c,p,idx,sites=None,trace=None):
    x=p['emb'][idx]
    if c.pos=='learned': x=x+p['pos'][None,:idx.shape[1]]
    if getattr(c,'ngram',0):
        before=j.where((j.arange(x.shape[1])>=1)[None,:,None],j.roll(x,1,axis=1),0.)
        x=x+lm.mm(lm.lin(p['ng_a'],x)*lm.lin(p['ng_b'],before),p['ng_o'])
    z=None
    for i,l in enumerate(p['layers']):
        layer={} if trace is not None else None
        stats={} if trace is not None else None
        a=attention(c,l,norm(c,l['ln1'],x),stats)
        if trace is not None: layer['stream']=x;layer['attn']=a;layer['stats']=stats
        before=x
        x=x+a+(sites[i]['attn'] if sites is not None else 0)
        if 'mg' in l: x=x+x*j.tanh(lm.mm(a,l['mg']))
        if 'local' in l: x=x+cubics.local(c,l['local'],x,None)
        if 'branch' in l:
            if c.arm=='pstream': b,z=cubics.pstream_step(c,l['branch'],x,z)
            else: b=branch(c,l['branch'],before if getattr(c,'parallel',0) else x,stats)
            b=cubics.residual(c,l['branch'],x,b)
            if trace is not None: layer['branch']=b
            x=x+b+(sites[i]['branch'] if sites is not None else 0)
        m=mlp(c,l,norm(c,l['ln2'],x))
        if trace is not None: layer['mlp']=m;trace.append(layer)
        x=x+m+(sites[i]['mlp'] if sites is not None else 0)
    return lm.mm(norm(c,p['lnf'],x),p['emb'].T)

def groups(p):
    """Parameter groups for gradient reporting."""
    out={'emb':{k:p[k] for k in ('emb','pos','ng_a','ng_b','ng_o') if k in p},'lnf':p['lnf']}
    for i,l in enumerate(p['layers']):
        out[f'attn{i}']={k:l[k] for k in ('ln1','qkv','o','mix_k2','mix_v2','mg') if k in l}
        out[f'mlp{i}']={k:l[k] for k in l if k in ('ln2','fc1','fc2','gate')}
        if 'branch' in l: out[f'branch{i}']=l['branch']
        if 'local' in l: out[f'local{i}']=l['local']
    return out

def counts(c,p):
    total=sum(a.size for a in jax.tree.leaves(p));branch_params=sum(a.size for l in p['layers'] for a in jax.tree.leaves(l.get('branch',{})))
    return dict(total=total,branch=branch_params,inactive=0,active=total)
