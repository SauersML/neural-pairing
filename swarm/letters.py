"""Which vocabulary tokens of a byte-level BPE tokenizer (Qwen3's) are letters, given the bytes still pending before
them. A token's bytes need not be whole characters: a newer emoji such as U+1F9F1 (F0 9F A7 B1) is two tokens, the
lead fragment F0 9F A7 and the lone continuation byte B1, and B1 alone also ends the letter U+00F1 (C3 B1). So
whether a token is a letter depends on what is pending: the unfinished character at the end of the tokens so far.

scan(bytes) reads bytes from a character boundary: they are a letter when they complete a character of category L*,
hold a byte that cannot be read (a continuation byte with no character open, an invalid byte, a character cut short
by a byte that does not continue it), or end in an unfinished character whose bytes occur inside some letter's UTF-8
encoding (letter_fragments, over all code points). Token t after pending bytes P is a letter when scan(P + t) is, and
leaves scan(P + t)'s unfinished character pending. With nothing pending, a token that starts with a continuation
byte is a letter.

LetterMasks(tok, V) holds the allowed tokens and one letter mask per distinct pending prefix (computed once each);
walk(ids) gives each token's letter flag and pending prefix in a message read from its start. Shared by
swarm/opentalk.py; usable by any trainer that pushes messages off letters."""
import unicodedata

import torch

_FRAGMENTS = []


def byte_chars():
    """Byte-level BPE's map from each byte to the character that stands for it in the vocabulary (GPT-2's)."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + \
        list(range(ord('\xae'), ord('\xff') + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def letter_fragments():
    """Every byte string that occurs inside the UTF-8 encoding of a letter (a character of category L*) without being
    the whole encoding, over all code points."""
    if not _FRAGMENTS:
        subs = set()
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF or not unicodedata.category(chr(cp)).startswith('L'):
                continue
            b = chr(cp).encode()
            subs.update(b[i:j] for i in range(len(b)) for j in range(i + 1, len(b) + 1) if j - i < len(b))
        _FRAGMENTS.append(subs)
    return _FRAGMENTS[0]


def _need(c):
    """A character's byte count from its first byte; 0 for a continuation byte or a byte UTF-8 never uses."""
    return 1 if c < 0x80 else 2 if 0xC2 <= c <= 0xDF else 3 if 0xE0 <= c <= 0xEF else 4 if 0xF0 <= c <= 0xF4 else 0


def scan(s, subs=None):
    """(letter, pending) for bytes s read from a character boundary (see the module docstring)."""
    subs = letter_fragments() if subs is None else subs
    i, letter = 0, False
    while i < len(s):
        n = _need(s[i])
        if n == 0:                                     # nothing open for it to continue, or not UTF-8 at all
            letter, i = True, i + 1
            continue
        j = i + 1
        while j < min(i + n, len(s)) and (s[j] & 0xC0) == 0x80:
            j += 1
        if j < min(i + n, len(s)):                     # cut short by a byte that does not continue it
            letter, i = True, j
            continue
        if i + n > len(s):                             # unfinished: the next token may end it
            tail = s[i:]
            return letter or tail in subs, tail
        try:
            ch = s[i:i + n].decode('utf-8')
        except UnicodeDecodeError:                     # overlong, surrogate or past U+10FFFF
            letter = True
        else:
            letter = letter or unicodedata.category(ch).startswith('L')
        i += n
    return letter, b''


def token_bytes(tok, V):
    """Each vocabulary row's bytes (None: special or added tokens, rows past the tokenizer; b'\\xff', never UTF-8, for
    a token with a character outside the byte map)."""
    dec = {c: b for b, c in byte_chars().items()}
    special = set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}))
    out = [None] * V
    for s, i in tok.get_vocab().items():
        if i < V and i not in special:
            try:
                out[i] = bytes(dec[c] for c in s)
            except KeyError:
                out[i] = b'\xff'
    return out


class LetterMasks:
    """allowed: bool [V], the tokens a message may use (not special or added, no control character but tab, newline
    or carriage return). mask(P): bool [V], the letters after pending bytes P (cached per P, on device). pending(P, t):
    the bytes pending after token t. walk(ids): (letter flags, pending prefix before each token) of a message."""

    def __init__(self, tok, V, device='cpu'):
        self.V, self.device, self.subs = V, device, letter_fragments()
        self.bytes = token_bytes(tok, V)
        allowed = torch.zeros(V, dtype=torch.bool)
        for i, b in enumerate(self.bytes):
            if b is None:
                continue
            try:
                text = b.decode('utf-8')
            except UnicodeDecodeError:
                allowed[i] = True                      # a fragment: allowed, a letter or not by context
                continue
            allowed[i] = bool(text) and not any(unicodedata.category(c) == 'Cc' and c not in '\t\n\r' for c in text)
        self.allowed = allowed.to(device)
        self.cont = [i for i, b in enumerate(self.bytes) if b and (b[0] & 0xC0) == 0x80]
        self._masks, self._cpu, self._pend = {}, {}, {}

    def mask(self, P=b''):
        if P not in self._masks:
            if P:                                      # only a token that starts by continuing P can avoid a letter
                m = torch.ones(self.V, dtype=torch.bool)
                for i in self.cont:
                    m[i] = scan(P + self.bytes[i], self.subs)[0]
            else:
                m = torch.tensor([True if b is None else scan(b, self.subs)[0] for b in self.bytes])
            self._cpu[P], self._masks[P] = m, m.to(self.device)
        return self._masks[P]

    def pending(self, P, t):
        key = (P, int(t))
        if key not in self._pend:
            b = self.bytes[int(t)]
            self._pend[key] = b'' if b is None else scan(P + b, self.subs)[1]
        return self._pend[key]

    def walk(self, ids):
        flags, pend, P = [], [], b''
        for t in ids:
            pend.append(P)
            self.mask(P)
            flags.append(bool(self._cpu[P][int(t)]))
            P = self.pending(P, t)
        return flags, pend
