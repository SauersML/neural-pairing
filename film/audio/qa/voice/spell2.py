"""Second spelling search on the dry formant voice: N and P articulation, syllable split."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import takes as T, asr
V = {
 "nlong":   ("N:160@1.4-UW1:200-R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1:200-R-IH0:110-NG:220."),
 "nsoft":   ("N:160@0.6-UW1:200-R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1:200-R-IH0:110-NG:220."),
 "nm":      ("M:40@0.5-N:140-UW1:200-R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1:200-R-IH0:110-NG:220."),
 "nd":      ("D:25@0.4-N:120-UW1:200-R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1:200-R-IH0:110-NG:220."),
 "nyuw":    ("N:140-Y:60-UW1:200-R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1:200-R-IY0:110-NG:220."),
 "ptpair":  ("N:140-UW1:200-R-AH0-L:100", "P:90@2-EH1:220-R:90-IH0:110-NG:240."),
}
SPLIT = {  # syllable per note: NEU | RAL | PAIR | ING
 "split":   ("N:140-UW1", "R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1", "R-IH0-NG:220."),
 "split_y": ("N:140-Y:60-UW1", "R-AH0-L:100", "P:80@1.8-HH:45@0.7-EH1", "R-IY0-NG:220."),
}
def song(lyr, inton, split=False, bpm=120):
    head = f"""bpm {bpm}
gate 0.95
tail 1.5
automate vox_mode
1
automate vox_level
0
automate vox_dry
1
automate vox_breath
0.1
automate vox_vibrato
0
automate vox_intonation
{inton}
track nv vox vel=0.9
"""
    if split:
        a, b, c, d = lyr
        return head + f"R:1 D3:0.75={a} D3:0.75={b} D3:1={c} D3:2={d}\n"
    return head + f"R:1 D3:1.5={lyr[0]} D3:2.5={lyr[1]}\n"
res = {}
for name, lyr in list(V.items()) + list(SPLIT.items()):
    for inton in (0.15, 0.6):
        n = f"dry2_{name}_i{int(inton*100)}"
        wp = T.render(n, song(lyr, inton, name in SPLIT))
        text, lp, ns = asr.transcribe(wp)
        sc = asr.score(text, "neural pairing")
        res[n] = (sc, text)
        print(f"{sc:5.2f} {n:<24} {text!r}", flush=True)
