"""Standalone cold-start invariants; CPU tiny architectures, no real model weights."""
import argparse
import math
import os
from pathlib import Path
import sys
os.environ['NL_DEVICE']='cpu'
os.environ['NL_TINY']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pairing'))
import torch
from transformers import AutoTokenizer
import link
from coldstart import add_arguments, bridge_groups, lr_multiplier, relay_episode
import session as S
import games as G
import json

torch.set_num_threads(2)
ap=argparse.ArgumentParser();add_arguments(ap);args=ap.parse_args([])
assert args.warmup_steps==100 and not args.gate_norm and not args.relay_first
for step in (1,20,100,500,1000):
    old=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*step/1000)))
    assert lr_multiplier(step,1000)==old
assert lr_multiplier(1,1000,0)>99*lr_multiplier(1,1000)

# Gate scale equivariance, including gradients to its weights and random-ablation gate ownership.
torch.manual_seed(22)
s=link.Side(128,1,2,16,[1.],gate_norm=True)
with torch.no_grad():s.gate[0].weight.normal_(0,.02)
h=torch.randn(2,4,128)
z=[]
for factor in (1.,.03):
    s.zero_grad(); gates=s.show(0,h*factor)[2]; gates.sum().backward()
    z.append((gates.detach(),s.gate[0].weight.grad.clone()))
assert torch.allclose(z[0][0],z[1][0],atol=2e-4)
assert torch.allclose(z[0][1],z[1][1],atol=.004)
assert torch.equal(s.show(0,h)[2],s.show(0,h,src=torch.randn_like(h))[2])
# Legacy raw gate retains its exact formula.
s.gate_norm=False
assert torch.equal(s.show(0,h)[2],torch.sigmoid(s.gate[0](h).squeeze(-1)))

# Identity starts: alpha=0 -> only alpha can learn; O=0, alpha=.5 -> full O learns.
for start in (0.,.5):
    r=link.Side(128,1,2,16,[.1]);sender=link.Side(128,1,2,16,[.03])
    with torch.no_grad():
        r.alpha.fill_(math.atanh(start))
        if start:r.o[0].weight.zero_()
    keys,vals,_=sender.show(0,h)
    w=r.read(0,h,keys,vals,torch.ones(2,4,4,dtype=torch.bool),torch.zeros(2,4,4,dtype=torch.long),torch.ones(2,4))
    assert torch.count_nonzero(w)==0
    (w*torch.randn_like(w)).sum().backward()
    assert (r.o[0].weight.grad.abs().sum()>0)==bool(start)
    assert (r.alpha.grad.abs().sum()>0)==(not start)
    assert r.q[0].weight.grad.abs().sum()==0
    assert sender.v[0].weight.grad.abs().sum()==0

# Real game/tokenizer path. No substituted payload metric.
root=Path(__file__).resolve().parents[1]
replies_path=os.environ.get('COLDSTART_REPLIES','/Users/user/mpd-data/neural-pairing/replies.json')
R=json.load(open(replies_path));g=G.Games(R,'train',seed=11)
toks={x:AutoTokenizer.from_pretrained(link.MODELS[x]['hf']) for x in link.SIDES}
for i in range(8):
    ep=relay_episode(g);r=ep['meta']['receiver'];s=ep['meta']['sender']
    js=S.payload_labels(ep,toks)[r];built=S.build(ep,toks)
    assert js and all(built[s]['labels'][j][2]==0 for j in range(len(built[s]['labels'])))
    assert all(built[r]['labels'][j][2]==1 for j in js)
    assert len(built[s]['labels'])==1
    assert ep['meta']['value'] not in ep['task'][r]

models={x:link.load(x)[1] for x in link.SIDES}
p=link.Pair(models,toks,(1,2,4),heads=2,head_dim=16,gate_norm=True)
ck=p.checkpoint();q=link.Pair.from_checkpoint(models,toks,ck)
assert all(s.gate_norm for s in q.bridge.values())
del ck['gate_norm'];q=link.Pair.from_checkpoint(models,toks,ck)
assert not any(s.gate_norm for s in q.bridge.values())
groups=bridge_groups(p,.001,.01,{'quill':None,'ember':.002})
assert [g['lr'] for g in groups]==[.001,.002,.01]
ids=[id(v) for g in groups for v in g['params']]
assert len(ids)==len(set(ids))==len(list(p.bridge.parameters()))
# --hold: the gates and alpha keep their exact values through an AdamW step; every other bridge parameter moves.
from coldstart import hold
for held in (True, False):
    torch.manual_seed(5)
    p=link.Pair(models,toks,(1,2,4),heads=2,head_dim=16,gate_norm=True)
    before={n:v.detach().clone() for n,v in p.bridge.named_parameters()}
    opt=torch.optim.AdamW(bridge_groups(p,.01,.01,{'quill':None,'ember':None}),weight_decay=0.0)
    sum((v*torch.randn_like(v)).sum() for v in p.bridge.parameters()).backward()
    if held:hold(p)
    opt.step()
    for n,v in p.bridge.named_parameters():
        frozen=n.split('.')[1] in ('alpha','gate')
        assert torch.equal(v,before[n])==(held and frozen),n
print('cold-start invariants passed')
