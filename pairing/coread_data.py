"""Content keys for the held-out split, shared by build_coread.py and coread.py.

A dataset's own split does not separate content: questions reuse contexts, and translation pairs recur in the other
direction or in another source. A test item is held out only when none of its content texts occurs in a training item.
Labels and answers are not content keys.

Usage: from coread_data import held_out; test = held_out(train, test)
"""
import hashlib
import re
import unicodedata


def text_key(text):
    """SHA-256 digest of text after NFKC normalization, casefolding and whitespace collapse."""
    text = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(text.encode()).digest()


def content_keys(item):
    """The text_keys of the item's content fields (for code_write also the problem with function names masked)."""
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
        # MBPP repeats a description with different function names.
        text = item['problem'].split(' It must pass:')[0]
        for name in re.findall(r'def\s+(\w+)', item['code']):
            text = re.sub(r'\b' + re.escape(name) + r'\b', 'FUNCTION', text)
        texts.append(text)
    return {text_key(t) for t in texts if t.strip()}


def held_out(train, test):
    """The test items none of whose content keys occur in a training item (exact match after normalization; near
    duplicates are kept)."""
    seen = {key for item in train for key in content_keys(item)}
    return [item for item in test if not seen.intersection(content_keys(item))]
