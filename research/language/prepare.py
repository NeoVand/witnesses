"""Language-model data, prepared once on Modal (CPU, no training). python prepare.py prints one JSON line.

tokenize(): FineWeb-Edu sample/10BT/000_00000.parquet as GPT-2 tokens (tiktoken), <|endoftext|> (50256) after every
document; every hundredth document is held out. Writes /vol/fineweb_edu/{train,val}.bin (uint16).
corpus(): the pinned held-out sets. Dev: 128 documents from the first half of the held-out stream; test: 1,024 from the
second half; documents of at least 32 tokens whose exact token sequence does not occur as a complete training
document, no two alike. Each row is a document's first 257 tokens (EOT-padded) with a mask over the 256 targets.
Writes /vol/data-v1/{dev,test}.npz and manifest.json (document indices, offsets, lengths, sha256; corpus hashes).
"""
import hashlib,json,os
from pathlib import Path
import numpy as np


def tokenize():
    import pyarrow.parquet as pq
    import tiktoken
    from huggingface_hub import hf_hub_download

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    out = "/vol/fineweb_edu"
    os.makedirs(out, exist_ok=True)
    counts = {}

    def write(docs_iter, path):
        n = 0
        with open(path, "wb") as fh:
            for docs in docs_iter:
                ids = enc.encode_ordinary_batch(docs, num_threads=16)
                arr = np.fromiter((t for d in ids for t in (*d, eot)), dtype=np.uint16)
                fh.write(arr.tobytes())
                n += len(arr)
        return n

    def chunks(seq, k=20000):
        for i in range(0, len(seq), k):
            yield seq[i : i + k]

    p = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", repo_type="dataset", cache_dir="/tmp/hf")
    pf = pq.ParquetFile(p)
    held = []

    def train_docs():
        i = 0
        for b in pf.iter_batches(batch_size=20000, columns=["text"]):
            docs = b.column(0).to_pylist()
            keep = []
            for d in docs:
                (held if i % 100 == 0 else keep).append(d)  # every hundredth document is held out
                i += 1
            yield keep

    counts["train"] = write(train_docs(), f"{out}/train.bin")
    counts["val"] = write(chunks(held), f"{out}/val.bin")
    return counts

def corpus():
    cache=Path('/vol/data-v1');cache.mkdir(parents=True,exist_ok=False)
    train=np.memmap('/vol/fineweb_edu/train.bin',dtype=np.uint16,mode='r')
    val=np.memmap('/vol/fineweb_edu/val.bin',dtype=np.uint16,mode='r')
    ve=np.flatnonzero(val==50256);vs=np.r_[0,ve[:-1]+1];vl=ve-vs
    candidates=np.flatnonzero(vl>=32)
    hashes={int(i):hashlib.sha256(val[vs[i]:ve[i]].tobytes()).hexdigest() for i in candidates}
    wanted=set(hashes.values());overlap=set();ntrain=0
    # Scan complete training documents, with bounded memory for boundary positions.
    start=0
    for off in range(0,len(train),10000000):
        ends=np.flatnonzero(train[off:off+10000000]==50256)+off
        for end in ends:
            if end-start>=32:
                h=hashlib.sha256(train[start:end].tobytes()).hexdigest()
                if h in wanted:overlap.add(h)
            start=int(end)+1;ntrain+=1
    rng=np.random.default_rng(17011);used=set();meta={}
    for name,n,side in [('dev',128,0),('test',1024,1)]:
        pool=[i for i in candidates if int(vs[i]>=len(val)//2)==side and hashes[int(i)] not in overlap]
        pool=list(rng.permutation(pool));chosen=[]
        for i in pool:
            if hashes[int(i)] in used:continue
            used.add(hashes[int(i)]);chosen.append(int(i))
            if len(chosen)==n:break
        assert len(chosen)==n, (name,len(chosen))
        tok=np.full((n,257),50256,np.int32);mask=np.zeros((n,256),np.float32)
        for row,i in enumerate(chosen):
            take=min(257,int(vl[i])+1);tok[row,:take]=val[vs[i]:vs[i]+take];mask[row,:take-1]=1
        np.savez(cache/(name+'.npz'),tokens=tok,mask=mask,document=np.array(chosen),offset=vs[chosen])
        meta[name]=[dict(document=i,offset=int(vs[i]),length=int(vl[i]),sha256=hashes[i]) for i in chosen]
    meta.update(train_tokens=len(train),val_tokens=len(val),complete_training_documents=ntrain,rejected_val_hashes_present_in_train=len(overlap),
                train_sha256=hashlib.file_digest(open('/vol/fineweb_edu/train.bin','rb'),'sha256').hexdigest(),
                val_sha256=hashlib.file_digest(open('/vol/fineweb_edu/val.bin','rb'),'sha256').hexdigest(),
                limitation='Exact full-document token hashes remove exact duplicate documents, not near duplicates or quoted passages. Upstream historical corpus revision unavailable, bytes pinned.')
    (cache/'manifest.json').write_text(json.dumps(meta,indent=2));return meta

if __name__=='__main__':print(json.dumps(dict(tokens=tokenize(),heldout=corpus())))
