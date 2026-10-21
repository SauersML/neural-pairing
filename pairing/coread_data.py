"""Content-level holdouts shared by the corpus builder and episode loader.

Rows alone are not a split boundary: questions reuse contexts and translation
pairs recur in another direction or source. Labels and answers are not split keys.
"""
import hashlib
import re
import unicodedata


def text_key(text):
    text = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(text.encode()).digest()


def content_keys(item):
    fields = {
        'passage': ('text',), 'qa': ('context', 'question'),
        'hop': ('a', 'b', 'question'), 'translate': ('source', 'target'),
        'paraphrase': ('a', 'b'), 'nli': ('premise', 'hypothesis'),
        'math': ('question',), 'code_out': ('code',),
        'code_write': ('problem', 'code'), 'instruct': ('instruction',),
        'mcq': ('question',),
    }[item['task']]
    texts = [item[f] for f in fields]
    if item['task'] == 'code_write':
        # MBPP repeats descriptions with different function signatures.
        text = item['problem'].split(' It must pass:')[0]
        for name in re.findall(r'def\s+(\w+)', item['code']):
            text = re.sub(r'\b' + re.escape(name) + r'\b', 'FUNCTION', text)
        texts.append(text)
    return {text_key(t) for t in texts if t.strip()}


def held_out(train, test):
    """Keep test rows with no complete content component seen in training.

    Exact normalized text decontamination, not semantic deduplication or a claim
    that the base models never saw the public datasets during pretraining.
    """
    seen = {key for item in train for key in content_keys(item)}
    return [item for item in test if not seen.intersection(content_keys(item))]
