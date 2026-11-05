"""A copy of scratch/lead/roleq/secrets_scan.py (the mechanistic_data release scanner). Finds text that holds a real credential or a person's local path (the standard of mechanistic_data PR #35): access
tokens and keys (fixed shapes, Telegram bot tokens, block-explorer keys, and a long random-looking value after a word
like token, key or secret), user:password@host URLs, a password next to a value, an IP address next to a login word, and
/Users/NAME, C:\\Users\\NAME or /home/NAME paths (generic names such as "user" or "ubuntu" are not counted).
  flags(text) -> sorted list of the kinds found
  python secrets_scan.py FILE.jsonl.gz [FIELD]   (counts per kind over the records)"""
import collections
import gzip
import json
import math
import re
import sys

TOKEN = re.compile(r'(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}'
                   r'|hf_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|[ps]k\.eyJ[A-Za-z0-9._-]{20,}'
                   r'|AIza[0-9A-Za-z_-]{35}|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+'
                   r'|-----BEGIN [A-Z ]*PRIVATE KEY-----|[sr]k_live_[A-Za-z0-9]{16,}|SG\.[\w-]{16,}\.[\w-]{16,}|glpat-[\w-]{16,}'
                   r'|npm_[A-Za-z0-9]{30,}|ya29\.[\w-]{20,}|hooks\.slack\.com/services/[A-Za-z0-9/]{20,}'
                   r'|discord(?:app)?\.com/api/webhooks/\d+/[\w-]{20,}|infura\.io/v3/[0-9a-f]{32}'
                   r'|alchemy\.com/v2/[\w-]{20,})')
TELEGRAM = re.compile(r'(?<![\w:])\d{6,12}:[A-Za-z0-9_-]{30,}(?![\w-])')  # bot id, colon, secret
EXPLORER = re.compile(r'(?<![A-Za-z0-9])[A-Z0-9]{34}(?![A-Za-z0-9])')  # BscScan, Etherscan and their kin
# a long value straight after one of these words, kept only if it looks random (random_looking)
AFTER_WORD = re.compile(r"""(?i)(?:api[_-]?key|apikey|token|secret|access[_-]?key|private[_-]?key|auth[_-]?key|bearer|\bkey)"""
                        r"""["'`‘’“”\]]*[ \t]*(?:=>|[:= \t])[ \t]*["'`‘’“”]?([A-Za-z0-9_\-+/=.]{20,})""")
PRODUCT_KEY = re.compile(r'(?<![A-Za-z0-9-])[A-Za-z0-9]{5}(?:-[A-Za-z0-9]{5}){4}(?![A-Za-z0-9-])')  # Windows and Office
PLACEHOLDER = ('your', 'example', 'xxxx', 'placeholder', 'dummy', 'sample', 'changeme', 'replace', 'abcdef', '123456', '0000')
KEYVAL = re.compile(r'(?i)(api[_-]?key|secret|token|access[_-]?key|client[_-]?secret)[ \t]*[:=][ \t]*["\'][A-Za-z0-9_\-./+]{16,}["\']')
USERPASS = re.compile(r'[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@"\']+:[^/\s@"\']+@')
PASSWORD = re.compile(r'(?i)\b(password|passwd|pwd|passphrase)\b["\']?[ \t]*[:=][ \t]*["\']?[^\s"\',)]{3,}')  # one line only
IP = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
LOGIN = re.compile(r'(?i)\b(login|username|password|passwd|ssh|sudo)\b')
PATH = re.compile(r'(?:/Users/|[A-Za-z]:(?:\\\\|\\|/)+Users(?:\\\\|\\|/)+|/home/)([A-Za-z0-9._ -]+)')
GENERIC = {'user', 'username', 'ubuntu', 'root', 'pi', 'admin', 'runner', 'your_username', 'yourusername', 'yourname',
           'your-username', 'name', 'me', 'public', 'shared', 'default', 'default user', 'all users', 'example',
           'your username', 'your_user', 'youruser', 'myuser', 'johndoe', 'john', 'jdoe', 'test', 'guest', 'vagrant',
           'ec2-user', 'centos', 'debian', 'node', 'app', 'www-data', 'git', 'docker', 'jovyan', 'linuxbrew'}


def entropy(v):
    """Shannon entropy of a string, in bits per character."""
    n = len(v)
    return -sum(c / n * math.log2(c / n) for c in collections.Counter(v).values())


def random_looking(v, floor=3.7):
    """Digits and letters mixed with high entropy, and no placeholder word: not something a person wrote as a name."""
    low = v.lower()
    if any(w in low for w in PLACEHOLDER):
        return False
    digits = sum(c.isdigit() for c in v)
    return digits >= 3 and sum(c.isalpha() for c in v) >= 6 and entropy(v) >= floor


def product_key(v):
    """Five groups of five with letters and digits mixed, and not a row of Xs or a counting pattern."""
    low = v.lower()
    bare = v.replace('-', '')
    return (not any(w in low for w in PLACEHOLDER) and any(c.isdigit() for c in bare) and any(c.isalpha() for c in bare)
            and entropy(bare) >= 2.5)


def values(text):
    """(kind, value) for the four shape checks that need the value itself: the wide scan's reports use these."""
    out = [('telegram bot token', m.group(0)) for m in TELEGRAM.finditer(text)]
    out += [('block explorer key', m.group(0)) for m in EXPLORER.finditer(text) if random_looking(m.group(0), 3.5)]
    out += [('high-entropy value', m.group(1)) for m in AFTER_WORD.finditer(text) if random_looking(m.group(1))]
    out += [('product key', m.group(0)) for m in PRODUCT_KEY.finditer(text) if product_key(m.group(0))]
    return out


def spans(text):
    """(kind, matched text, position) for every check: flags() is exactly the set of these kinds."""
    for k, v in values(text):
        yield k, v, text.find(v)
    for m in TOKEN.finditer(text):
        yield 'token', m.group(0), m.start()
    for m in KEYVAL.finditer(text):
        yield 'key value', m.group(0), m.start()
    for m in USERPASS.finditer(text):
        yield 'user:password@host', m.group(0), m.start()
    for m in PASSWORD.finditer(text):
        yield 'password value', m.group(0), m.start()
    if LOGIN.search(text):
        for m in IP.finditer(text):
            yield 'ip with login word', m.group(0), m.start()
    for m in PATH.finditer(text):
        name = m.group(1).strip().split('/')[0].split('\\')[0].strip().lower()
        if name and name not in GENERIC and not name.startswith('<') and not name.startswith('$'):
            yield 'user path', m.group(0), m.start()


def flags(text):
    return sorted({k for k, _, _ in spans(text)})


def strings(obj):
    """Every string inside a JSON value, joined by newlines (scan raw text, not its JSON escaping)."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return '\n'.join(strings(v) for v in obj.values())
    if isinstance(obj, list):
        return '\n'.join(strings(v) for v in obj)
    return ''


def main(path, field=None):
    kinds = collections.Counter()
    n = 0
    with gzip.open(path, 'rt') as f:
        for line in f:
            r = json.loads(line)
            n += 1
            for k in flags(strings(r.get(field) if field else r)):
                kinds[k] += 1
    print(n, dict(kinds))


if __name__ == '__main__':
    main(*sys.argv[1:3])
