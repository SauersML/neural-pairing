"""CPU random-token transfer benchmark (standalone, not pytest).

Uses link.load's NL_TINY Qwen/AFM architectures (six blocks, d=128); replaces only the vocabulary with its first
64 rows. Both bases and LM heads are frozen. Each side holds an independent uniform secret (IDs 8..23) in its prompt
and must predict the partner's secret, with no natural-language skill. Full 64-way CE, not candidate-restricted CE.
Fresh random distractors/locations every batch; fixed held-out prompt pairs explicitly excluded from training.
Teacher-forced filler labels after
the secret deliberately dilute the transfer objective. Evaluation always measures only the first, secret label.
'open' means both directions beat BOTH cut and isolated shuffled partners by >0.1 nat and >2 paired standard errors,
at two successive evaluations; reports the first of those two steps. Three seeds by default, fixed bases across arms.

Example: MPD_MEM_GIB=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
    /Users/user/mpd-data/venv/bin/python tests/coldstart_bench.py --out /tmp/coldstart.json
This is a mechanism experiment, not a claim of convergence speed for pretrained 4B models.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import torch
import torch.nn.functional as F
import link
import session as S
from coldstart import bridge_groups, lr_multiplier

VARIANTS = {
    'pure_baseline': dict(start=0., warmup=100, payload_w=1., tail=0, prompt=3, norm=False),
    'pure_read': dict(start=.5, warmup=100, payload_w=1., tail=0, prompt=3, norm=False),
    'pure_fast': dict(start=.5, warmup=10, payload_w=1., tail=0, prompt=3, norm=False),
    'pure_norm': dict(start=.5, warmup=10, payload_w=1., tail=0, prompt=3, norm=True),
    'pure_qk': dict(start=.5, warmup=10, payload_w=1., tail=0, prompt=3, norm=True, content=True),
    'baseline': dict(start=0., warmup=100, payload_w=1., tail=7, prompt=12, norm=False),
    'read_start': dict(start=.5, warmup=100, payload_w=1., tail=7, prompt=12, norm=False),
    'short_warmup': dict(start=.5, warmup=10, payload_w=1., tail=7, prompt=12, norm=False),
    'payload16': dict(start=.5, warmup=10, payload_w=16., tail=7, prompt=12, norm=False),
    'gate_norm': dict(start=.5, warmup=10, payload_w=16., tail=7, prompt=12, norm=True),
    'relay_only': dict(start=.5, warmup=10, payload_w=1., tail=0, prompt=6, norm=False),
    'relay_norm': dict(start=.5, warmup=10, payload_w=1., tail=0, prompt=6, norm=True),
    'alpha_fast': dict(start=0., warmup=100, payload_w=1., tail=7, prompt=12, norm=False, alpha_lr=.03),
    'content_qk': dict(start=.5, warmup=10, payload_w=16., tail=7, prompt=12, norm=True, content=True),
    'ember_fast': dict(start=.5, warmup=10, payload_w=16., tail=7, prompt=12, norm=True, ember_lr=.0012),
}


def episodes(gen, bs, prompt=12, tail=7):
    secrets = torch.randint(8, 24, (bs, 2), generator=gen)
    built = []
    for i in range(bs):
        ep = {}
        for j, x in enumerate(link.SIDES):
            ids = torch.randint(32, 64, (prompt,), generator=gen).tolist()
            ids[int(torch.randint(1, prompt - 1, (), generator=gen))] = int(secrets[i, j])
            ids[-1] = 2
            # The secret target NEVER enters the stream until after its prediction position.
            target = [int(secrets[i, 1-j])] + [3 + k % 4 for k in range(tail)]
            seq = ids + target[:-1]
            ep[x] = dict(ids=seq, time=[-1]*(prompt-1)+list(range(len(target))), cut=None,
                         labels=[(prompt-1+k, t, 1.) for k,t in enumerate(target)])
        built.append(ep)
    return built


def prompt_key(ep):
    return tuple(tuple(t for t,tick in zip(ep[x]['ids'],ep[x]['time']) if tick<=0) for x in link.SIDES)


def batch(gen, bs, cfg, exclude=None):
    built=episodes(gen, bs, cfg['prompt'], cfg['tail'])
    if exclude:
        for i,ep in enumerate(built):
            while prompt_key(ep) in exclude:
                ep=episodes(gen,1,cfg['prompt'],cfg['tail'])[0]
            built[i]=ep
    return S.collate(built, {x:0 for x in link.SIDES})


def losses(pair, b, mode=None):
    last, _ = pair.joint(b, mode=mode)
    return {x:F.cross_entropy(pair.token_logits(x,last[x],b[x]['idx']),b[x]['lab'],reduction='none')
            .reshape(b[x]['ids'].shape[0],-1) for x in link.SIDES}


@torch.no_grad()
def evaluate(pair, held, cache=None):
    ces={mode:{x:[] for x in link.SIDES} for mode in ('link','zero','shuffle')}
    hits={mode:{x:[] for x in link.SIDES} for mode in ces}
    for bi,b in enumerate(held):
        for mode in ces:
            if mode == "zero" and cache is not None and bi in cache:
                for x in link.SIDES:
                    ces[mode][x].append(cache[bi][x][0]);hits[mode][x].append(cache[bi][x][1])
                continue
            last,_=pair.joint(b, mode=None if mode=='link' else mode)
            for x in link.SIDES:
                n=b[x]['ids'].shape[0]
                ix=b[x]['idx'].reshape(n,-1)[:,0]; target=b[x]['lab'].reshape(n,-1)[:,0]
                logits=pair.token_logits(x,last[x],ix)
                ces[mode][x].append(F.cross_entropy(logits,target,reduction='none'))
                hits[mode][x].append((logits.argmax(-1)==target).float())
            if mode == 'zero' and cache is not None:
                cache[bi]={x:(ces[mode][x][-1],hits[mode][x][-1]) for x in link.SIDES}
    ces={m:{x:torch.cat(v) for x,v in z.items()} for m,z in ces.items()}
    out={'ce':{m:{x:float(v.mean()) for x,v in z.items()} for m,z in ces.items()},
         'acc':{m:{x:float(torch.cat(v).mean()) for x,v in z.items()} for m,z in hits.items()}, 'open':True}
    out['delta']={}
    for x in link.SIDES:
        for mode in ('zero','shuffle'):
            d=ces['link'][x]-ces[mode][x]; mean=float(d.mean()); se=float(d.std()/math.sqrt(d.numel()))
            out['delta'][x+'/'+mode]=[mean,se]
            out['open'] &= mean < -max(.1,2*se)
    return out


@torch.no_grad()
def content_qk(pair, path):
    """Teammate align.fit/match on content-matched random token streams; no task labels used.

    Uses the exact ridge selection, mean removal and temperature selection of align.content_init, with numerical
    token IDs (our vocabulary is 64) instead of tokenizer character offsets. Evaluation episodes are excluded.
    """
    if path:
        sys.path.insert(0,path)
    import align
    import numpy as np
    gen=torch.Generator().manual_seed(7001)
    streams=[torch.randint(1,64,(12,),generator=gen).tolist() for _ in range(40)]
    def states_of(x,t):
        ids=torch.tensor([t]);r=link.Run(pair.models[x],ids,torch.ones_like(ids),torch.arange(len(t))[None])
        out=[]
        for k in range(pair.depth[x]):
            r.block(k)
            if k in pair.xblocks:
                h=r.h[0].float();out.append(h*torch.rsqrt(h.square().mean(-1,keepdim=True)+1e-6))
        return out
    states=[{x:states_of(x,t) for x in link.SIDES} for t in streams]
    test=[dict(t,pos={x:torch.arange(12) for x in link.SIDES}) for t in states[-8:]]
    out=[]
    for k in range(len(pair.xblocks)):
        for x in link.SIDES:
            y=link.partner(x);rx=pair.bridge[x];sy=pair.bridge[y]
            def stats(rows):
                a=torch.cat([t[x][k] for t in rows]).double().numpy()
                b=torch.cat([t[y][k] for t in rows]).double().numpy()
                return dict(xx=a.T@a,yy=b.T@b,xy=a.T@b,sx=a.sum(0),sy=b.sum(0),n=len(a))
            M,mx,r2,lam=align.fit(stats(states[:-8]),stats(states[-8:]),np.ones(128),np.ones(128))
            u=mx/np.linalg.norm(mx);Q=rx.q[k].weight.double().numpy();Q=Q-np.outer(Q@u,u)
            rx.q[k].weight.copy_(torch.from_numpy(Q));sy.k[k].weight.copy_(torch.from_numpy(Q@M))
            lp,top,chance=align.match(pair,x,y,k,test,align.SCALES);j=int(np.argmax(lp))
            sy.k[k].weight.mul_(float(align.SCALES[j]))
            out.append(dict(reader=x,block=k,r2=float(r2),top1=float(top[j]),scale=float(align.SCALES[j])))
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=300)
    ap.add_argument('--bs',type=int,default=4)
    ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--variants',default='pure_baseline,pure_read,pure_fast,pure_norm,pure_qk')
    ap.add_argument('--every',type=int,default=20)
    ap.add_argument('--eval_n',type=int,default=128)
    ap.add_argument('--eval_prompt',type=int,default=3,help='held-out prompt length, shared across arms')
    ap.add_argument('--lr_bridge',type=float,default=3e-4)
    ap.add_argument('--resume',action='store_true',help='skip complete runs already saved in --out')
    ap.add_argument('--stop_on_open',action='store_true')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--align_path',default=None,help='optional path containing teammate align.py; enables content_qk arm')
    args=ap.parse_args()
    if args.eval_n < 16 or args.eval_n % 16 or args.eval_prompt < 3 or min(args.bs,args.steps,args.every,args.threads) < 1:
        ap.error('positive steps/batch/evaluation interval/threads and --eval_n a positive multiple of 16 required')
    torch.set_num_threads(args.threads)
    toks,models={},{}
    for x in link.SIDES:
        toks[x],models[x]=link.load(x)
        # Slice exactly; no new model architecture or randomly trained classifier is substituted.
        m=models[x]; dec,head=link.inner(m)
        dec.embed_tokens=torch.nn.Embedding.from_pretrained(dec.embed_tokens.weight[:64].detach().clone())
        head.weight=torch.nn.Parameter(head.weight[:64].detach().clone(),requires_grad=False)
        head.out_features=64; m.config.vocab_size=64
        m.requires_grad_(False);m.eval()
    xb=(1,2,4)
    calibration=episodes(torch.Generator().manual_seed(987),8)
    sizes={x:link.residual_rms(models[x],xb,[e[x]['ids'] for e in calibration]) for x in link.SIDES}
    results={'args':vars(args),'sizes':sizes,'runs':[],'torch':torch.__version__}
    if args.resume and Path(args.out).exists():
        previous=json.loads(Path(args.out).read_text())
        for key in ('steps','bs','every','eval_n'):
            assert previous['args'][key]==vars(args)[key], key
        assert previous['args'].get('eval_prompt',12)==args.eval_prompt
        assert previous['args'].get('lr_bridge',3e-4)==args.lr_bridge
        results['runs']=previous['runs']
    done={(r['variant'],r['seed']) for r in results['runs']}
    cut_cache={}  # Bases are frozen, and every arm evaluates exactly the same episodes.
    for name in args.variants.split(','):
        cfg=VARIANTS[name]
        for seed in map(int,args.seeds.split(',')):
            if (name,seed) in done:continue
            torch.manual_seed(seed+100)
            pair=link.Pair(models,toks,xb,sizes,heads=2,head_dim=16,gate_norm=cfg['norm'])
            if cfg['start']:
                with torch.no_grad():
                    for s in pair.bridge.values():
                        s.alpha.fill_(math.atanh(cfg['start']))
                        for o in s.o:o.weight.zero_()
            alignment=content_qk(pair,args.align_path) if cfg.get('content') else None
            opt=torch.optim.AdamW(bridge_groups(pair,args.lr_bridge,cfg.get('alpha_lr',3e-3),
                         {'quill':None,'ember':cfg.get('ember_lr')}),betas=(.9,.98),weight_decay=0.)
            base=[g['lr'] for g in opt.param_groups]
            gen=torch.Generator().manual_seed(seed+200)
            eg=torch.Generator().manual_seed(4242)
            # Same held-out task for every arm. Future targets cannot influence the first prediction.
            held=[batch(eg,min(16,args.eval_n-i),dict(VARIANTS['baseline'],tail=0,prompt=args.eval_prompt))
                  for i in range(0,args.eval_n,16)]
            held_keys={tuple(tuple(b[x]['ids'][i].tolist()) for x in link.SIDES)
                       for b in held for i in range(b[link.SIDES[0]]['ids'].shape[0])}
            run={'variant':name,'seed':seed,'cfg':cfg,'history':[],'open_step':None,
                 'alignment':alignment,'disjoint_heldout':True}; prev=None;t=time.time()
            for step in range(args.steps+1):
                if step:
                    b=batch(gen,args.bs,cfg,held_keys);opt.zero_grad(set_to_none=True)
                    ce=losses(pair,b); loss=0
                    for x,v in ce.items():
                        weights=torch.ones(v.shape[-1]);weights[0]=cfg['payload_w']
                        loss=loss+(v*weights).sum()/(v.shape[0]*weights.sum())
                    loss.backward();torch.nn.utils.clip_grad_norm_(pair.bridge.parameters(),1.)
                    for g,lr in zip(opt.param_groups,base):g['lr']=lr*lr_multiplier(step,1000,cfg['warmup'])
                    opt.step()
                if step%args.every==0 or step==args.steps:
                    ev=evaluate(pair,held,cut_cache);ev['step']=step;run['history'].append(ev)
                    if ev['open'] and prev is not None and run['open_step'] is None:run['open_step']=prev
                    prev=step if ev['open'] else None
                    print(json.dumps({'variant':name,'seed':seed,'step':step,'open_step':run['open_step'],
                                      'ce':ev['ce'],'seconds':round(time.time()-t,1)}),flush=True)
                    if args.stop_on_open and run['open_step'] is not None:break
            run['seconds']=time.time()-t;results['runs'].append(run)
            results['disjoint_heldout']=all(r.get('disjoint_heldout',False) for r in results['runs'])
            Path(args.out).write_text(json.dumps(results,indent=2))
            del pair,opt


if __name__=='__main__':main()
