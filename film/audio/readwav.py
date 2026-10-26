import numpy as np, struct
def readwav(path):
    d = open(path,'rb').read()
    i = 12; fmt=None
    while i < len(d):
        cid = d[i:i+4]; sz = struct.unpack('<I', d[i+4:i+8])[0]
        if cid == b'fmt ':
            tag, ch, sr, _, _, bits = struct.unpack('<HHIIHH', d[i+8:i+24]); fmt=(tag,ch,sr,bits)
        elif cid == b'data':
            raw = d[i+8:i+8+sz]; break
        i += 8 + sz + (sz & 1)
    tag,ch,sr,bits = fmt
    if tag == 3 or (tag==0xFFFE and bits==32 and False):
        x = np.frombuffer(raw, '<f4')
    elif bits == 16:
        x = np.frombuffer(raw, '<i2').astype(np.float32)/32768
    elif bits == 24:
        b = np.frombuffer(raw, np.uint8).reshape(-1,3)
        v = (b[:,0].astype(np.int32) | (b[:,1].astype(np.int32)<<8) | (b[:,2].astype(np.int32)<<16))
        v = np.where(v >= 1<<23, v - (1<<24), v); x = v.astype(np.float32)/(1<<23)
    elif bits == 32:
        x = np.frombuffer(raw, '<f4') if tag in (3,0xFFFE) else np.frombuffer(raw,'<i4').astype(np.float32)/2**31
    return x.reshape(-1, ch), sr
