# Write a copy of a song keeping only the named tracks (plus all global lines):
#   python3 solo.py score.song out.song pad,bass
import sys, re
src, dst, keep = sys.argv[1], sys.argv[2], set(sys.argv[3].split(','))
out, cur_keep = [], True
for line in open(src):
    s = line.strip()
    m = re.match(r'track\s+(\S+)', s)
    a = re.match(r'automate\s+([^\s:]+)', s)
    if m:
        cur_keep = m.group(1) in keep
    elif a:
        name = a.group(1)
        cur_keep = (name.split('.')[0] in keep) if '.' in name else True
    elif re.match(r'(bpm|gate|tail|section)\b', s):
        cur_keep = True
    if cur_keep:
        out.append(line)
open(dst, 'w').writelines(out)
