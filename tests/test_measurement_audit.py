"""Measurement regressions with real tiny models and tokenizers; no replaced forward paths."""
import copy
import os
from pathlib import Path
import sys
import unittest
os.environ['NL_TINY'] = '1'
os.environ['NL_DEVICE'] = 'cpu'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pairing'))
import torch
import coread
import evaluate
import link
import payload_ctl
import play_tasks
import session as S


class MeasurementAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.toks, models = {}, {}
        for x in link.SIDES:
            cls.toks[x], models[x] = link.load(x)
            models[x].eval()
        cls.pair = link.Pair(models, cls.toks, (1, 2, 4), heads=2, head_dim=16).eval()
        for side in cls.pair.bridge.values():
            side.alpha.data.fill_(.7)

    def episodes(self):
        return [dict(game='relay_word', mode='duplex', system={x: f'You are {x}.' for x in link.SIDES},
                     task={'quill': 'Name the private word.', 'ember': f'The word is {w}.'},
                     reply={'quill': (f'The word is {w}.', 1.), 'ember': ('Ready.', 0.)},
                     meta=dict(receiver='quill', sender='ember', value=w)) for w in ('fox', 'hippopotamus')]

    def test_table_rejects_mixed_games(self):
        eps = self.episodes()
        eps[1]['game'] = 'lookup'
        with self.assertRaisesRegex(ValueError, 'one game'):
            payload_ctl.table(self.pair, self.toks, eps)

    def test_mcq_whole_option(self):
        item = dict(options=['cat', 'dog'], answer=0)
        for text in ('category', 'bobcat', 'catfish'):
            self.assertEqual(play_tasks.score_one('co_mcq', item, text, 'quill', None), 0.)
        self.assertEqual(play_tasks.score_one('co_mcq', item, 'The cat.', 'quill', None), 1.)
        item = dict(options=['cat', 'catfish'], answer=1)
        self.assertEqual(play_tasks.score_one('co_mcq', item, 'catfish', 'quill', None), 1.)

    def test_play_condition_keeps_each_receiver(self):
        eps=[]
        for opener in link.SIDES:
            secrets=dict(quill={'word':'fox'}, ember={'word':'otter'})
            ep=dict(game='co_play',mode='dialogue',system={x:coread.play_system('guess',x,opener,secrets) for x in link.SIDES},
                    task={x:None for x in link.SIDES}, turns=[(opener,'otter',1.),(link.partner(opener),'fox',1.)],
                    meta=dict(opener=opener,secrets=secrets,sender=link.partner(opener),receiver=opener))
            eps.append(ep)
        old=os.environ.get('NL_MAXNEW');os.environ['NL_MAXNEW']='1'
        try:
            row=evaluate.condition(self.pair,eps,'co_play','link',2)
        finally:
            if old is None: os.environ.pop('NL_MAXNEW')
            else: os.environ['NL_MAXNEW']=old
        self.assertEqual(set(row['each']),set(link.SIDES))
        for x in link.SIDES:
            self.assertEqual(row['each'][x],evaluate.mean([r['each'][x] for r in row['examples']]))

    def test_table_pairing_and_broken_labels(self):
        eps=self.episodes()
        good=payload_ctl.table(self.pair,self.toks,eps,bs=2)
        bad=copy.deepcopy(eps)
        for i in range(2):
            bad[i]['reply']['quill']=eps[1-i]['reply']['quill']
            bad[i]['meta']['value']=eps[1-i]['meta']['value']
        wrong=payload_ctl.table(self.pair,self.toks,bad,bs=2)
        self.assertGreater(max(abs(a['link_ce']-b['link_ce']) for a,b in zip(good,wrong)),1e-4)
        for i, correct in enumerate(good):
            swapped = wrong[1-i]
            self.assertAlmostEqual(correct['link_ce'], swapped['shuffle_ce'], places=5)
            self.assertAlmostEqual(correct['shuffle_ce'], swapped['link_ce'], places=5)
        for episodes,records in ((eps,good),(bad,wrong),(list(reversed(eps)),payload_ctl.table(self.pair,self.toks,list(reversed(eps)),bs=2))):
            built=[S.build(ep,self.toks) for ep in episodes]
            b=S.collate(built,{x:self.toks[x].pad_token_id for x in link.SIDES})
            for name,mode in payload_ctl.CONDS:
                with torch.no_grad(): h,_=self.pair.joint(b,mode=mode)
                for i,ep in enumerate(episodes):
                    selected=[built[i]['quill']['labels'][j] for j in S.payload_labels(ep,self.toks)['quill']]
                    pos=torch.tensor([p for p,_,_ in selected]); target=torch.tensor([t for _,t,_ in selected])
                    logits=link.inner(self.pair.models['quill'])[1](h['quill'][i,pos])
                    expected=float(torch.nn.functional.cross_entropy(logits,target).detach())
                    self.assertAlmostEqual(records[i][name+'_ce'],expected,places=5)
        print('real tiny table good:',good,'swapped-label table:',wrong,flush=True)

if __name__=='__main__': unittest.main(verbosity=2)
