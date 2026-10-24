"""A pointwise judge: a local open model's teacher-forced score of one text on a rubric. No API, no sampling.

Judge(hf, device, bs, think) loads hf in bf16 on one device (DEVICE, a CUDA index or a device name), never split
over GPUs: on the pods' multi-A40 hosts direct GPU-to-GPU copies are corrupt (a 4096 x 4096 copy from GPU 0 to GPU 1
differed by up to 5.5), which gave Qwen3-32B split by device_map the same logit for every token and Mistral-Small
non-finite logits.
score(rubric, texts) shows the judge each text after the rubric, asks for one digit from 1 to 9, and reads its
next-token probabilities after its chat template's assistant header (thinking off where the template has it): the
score is the expected digit over the nine digit tokens (each must be one token), returned with the probability the
judge put on digits at all (check() refuses a judge that puts almost none there, or does not complete "The capital
of France is" with Paris). With think > 0 the judge first reasons greedily, thinking on, for up to think tokens (an
unclosed reasoning is closed there; eager attention, and any non-finite logit is an error), and the digit is read
after the reasoning. Scoring batches are right-padded and read at each text's last real token, so no real position
attends to padding; any non-finite score is an error, and check() compares batched with one-at-a-time scores.

Two texts compare by their score difference: strict_win(d, margin) is 1 when it is above margin, 0 when it is below
-margin, 1/2 between. Judges favour longer text, so dialogues are compared length-matched (matched): each turn of the
two is cut to the shorter one's words.

NL_TINY=1: link.tiny's random model of hf's architecture, on the CPU (smoke tests).
"""
import os

import torch

from link import DEVICE, tiny

TEMPLATE = '{rubric}\n\nConversation:\n{text}\n\nAnswer with a single digit from 1 to 9.'
DIGITS = '123456789'


class Finite:
    """A generation logits processor that refuses non-finite logits."""

    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise FloatingPointError('non-finite logits while reasoning')
        return scores


class Judge:
    def __init__(self, hf, device=None, bs=8, think=0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.hf, self.bs, self.think = hf, bs, think
        self.label = hf + (f'+think{think}' if think else '')
        # Mistral's tokenizer files carry a wrong pre-tokenizer regex that transformers repairs on request
        self.tok = AutoTokenizer.from_pretrained(hf, **({'fix_mistral_regex': True} if 'mistral' in hf.lower() else {}))
        kw = dict(dtype=torch.bfloat16, **({'attn_implementation': 'eager'} if think else {}))
        device = DEVICE if device is None else device if isinstance(device, str) else f'cuda:{device}'
        self.model = (tiny(hf) if os.environ.get('NL_TINY') else AutoModelForCausalLM.from_pretrained(hf, **kw)).to(device)
        self.model.eval().requires_grad_(False)
        self.device = self.model.get_input_embeddings().weight.device
        self.digits = []
        for c in DIGITS:
            t = self.tok(c, add_special_tokens=False).input_ids
            if len(t) != 1:
                raise ValueError(f'{hf}: {c!r} is not one token')
            self.digits.append(t[0])
        self.values = torch.arange(1, 10, dtype=torch.float32)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def prompt(self, rubric, text, think=None):
        msgs = [dict(role='user', content=TEMPLATE.format(rubric=rubric, text=text))]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=bool(self.think if think is None else think))

    def ids(self, rubric, texts, think=None):
        return [self.tok(self.prompt(rubric, t, think), add_special_tokens=False).input_ids for t in texts]

    @torch.no_grad()
    def last_logits(self, seqs):
        """The next-token logits after each token sequence (float32, CPU): right-padded batches in length order."""
        body, head = self.model.get_decoder(), self.model.get_output_embeddings()
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = [None] * len(seqs)
        for k in range(0, len(order), self.bs):
            idx = order[k:k + self.bs]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.tensor([seqs[i] + [self.pad] * (L - len(seqs[i])) for i in idx], device=self.device)
            att = torch.tensor([[1] * len(seqs[i]) + [0] * (L - len(seqs[i])) for i in idx], device=self.device)
            h = body(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
            last = torch.tensor([len(seqs[i]) - 1 for i in idx], device=h.device)
            h = h[torch.arange(len(idx), device=h.device), last]
            logits = head(h.to(head.weight.device)).float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'{self.label}: non-finite logits')
            for i, l in zip(idx, logits):
                out[i] = l
        return torch.stack(out)

    def read(self, seqs):
        """(expected digit, probability on the digits) after each token sequence."""
        logits = self.last_logits(seqs)
        d = logits[:, self.digits]
        score = d.softmax(-1) @ self.values
        mass = (d.logsumexp(-1) - logits.logsumexp(-1)).exp()
        return score.tolist(), mass.tolist()

    @torch.no_grad()
    def reason(self, seqs):
        """Each prompt followed by the judge's greedy reasoning, closed, ending where the answer starts."""
        close = self.tok('</think>', add_special_tokens=False).input_ids
        if len(close) != 1:
            raise ValueError(f'{self.hf}: no single </think> token')
        after, closing = (self.tok(s, add_special_tokens=False).input_ids for s in ('\n\n', '\n</think>\n\n'))
        stops = {close[0], self.pad, *([self.tok.eos_token_id] if self.tok.eos_token_id is not None else [])}
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = [None] * len(seqs)
        for k in range(0, len(order), self.bs):
            idx = order[k:k + self.bs]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.tensor([[self.pad] * (L - len(seqs[i])) + seqs[i] for i in idx], device=self.device)
            att = torch.tensor([[0] * (L - len(seqs[i])) + [1] * len(seqs[i]) for i in idx], device=self.device)
            gen = self.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=self.think, do_sample=False,
                                      eos_token_id=sorted(stops), pad_token_id=self.pad, logits_processor=[Finite()])
            for i, g in zip(idx, gen[:, L:].tolist()):
                stop = next((j for j, t in enumerate(g) if t in stops), len(g))
                closed = stop < len(g) and g[stop] == close[0]
                out[i] = seqs[i] + g[:stop] + (close + after if closed else closing)
        return out

    def score(self, rubric, texts):
        """([score], [probability on the digits]) for each text on the rubric."""
        seqs = self.ids(rubric, texts)
        return self.read(self.reason(seqs) if self.think else seqs)

    def france(self):
        """The five likeliest next tokens after "The capital of France is", decoded."""
        bos = [] if self.tok.bos_token_id is None else [self.tok.bos_token_id]
        ids = bos + self.tok('The capital of France is', add_special_tokens=False).input_ids
        return [self.tok.decode([t]) for t in self.last_logits([ids])[0].topk(5).indices.tolist()]

    def check(self, rubric, texts, tol=0.5, least=None):
        """(The largest gap between each text's score alone and inside one padded batch, the mean probability on the
        digits, the France completion), thinking off; an error when the gap is above tol points (bf16 batch noise was
        0.10 for Qwen3-8B on the Mac GPU and 0.28 on an A40; a padding bug moves scores by more), the probability is
        below least (a broken model's is about 9 over its vocabulary) or Paris is not among the five likeliest
        completions (no floors for NL_TINY's random models)."""
        tiny_ = bool(os.environ.get('NL_TINY'))
        least = (0.0 if tiny_ else 0.05) if least is None else least
        top = self.france()
        if not tiny_ and not any(t.strip() == 'Paris' for t in top):
            raise FloatingPointError(f'{self.label}: "The capital of France is" -> {top}; the model is broken')
        seqs = self.ids(rubric, texts, think=False)
        bs, self.bs = self.bs, 1
        alone, mass = self.read(seqs)
        self.bs = len(seqs)
        together, _ = self.read(seqs)
        self.bs = bs
        gap, mass = max(abs(a - b) for a, b in zip(alone, together)), sum(mass) / len(mass)
        if gap > tol:
            raise FloatingPointError(f'{self.label}: batched scores differ from single ones by {gap:.3f}')
        if mass < least:
            raise FloatingPointError(f'{self.label}: probability {mass:.2g} on the digits; the model is broken')
        return gap, mass, top


def strict_win(d, margin):
    """A score difference as a strict win: 1 above margin, 0 below -margin, 1/2 between."""
    return 1.0 if d > margin else 0.0 if d < -margin else 0.5


def transcript(turns, names):
    """A dialogue's turns [(speaker, text)] as lines "Name: text"."""
    return '\n'.join(f'{names[x]}: {text.strip()}' for x, text in turns)


def matched(a, b, names):
    """Two dialogues of the same turn order as transcripts, each turn cut to the shorter one's words (marked ...)."""
    if [x for x, _ in a] != [x for x, _ in b]:
        raise ValueError('length matching needs the same speakers in the same order')
    ta, tb = [], []
    for (x, s), (_, t) in zip(a, b):
        u, v = s.split(), t.split()
        k = min(len(u), len(v))
        ta.append((x, ' '.join(u[:k]) + (' ...' if len(u) > k else '')))
        tb.append((x, ' '.join(v[:k]) + (' ...' if len(v) > k else '')))
    return transcript(ta, names), transcript(tb, names)
