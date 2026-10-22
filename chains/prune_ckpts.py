"""Keeps a running pod's disk from filling with checkpoints: for each run named, deletes on the pod every step_N folder
of its run/ and warm/ outputs except the newest 3 of each, and the one the chain will evaluate, once every file of the
Mac's copy has the same sha256 as the pod's. Repeats every 10 minutes until the runs end.
Usage: python prune_ckpts.py magic-nl20 magic-nl21 ..."""
import hashlib, os, re, subprocess, sys, time

RUNPOD = '/Users/user/mpd-data/runpod'
PODCP = '/Users/user/mpd-data/neural-pairing/podcp.sh'
KEEP = 3


def on_pod(name, cmd):
    r = subprocess.run(['bash', PODCP, name, '--run', cmd], capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    if r.returncode:
        raise RuntimeError(f'{name}: {cmd[:60]}: exit {r.returncode}')
    return r.stdout


def mac_hashes(folder):
    out = {}
    for d, _, fs in os.walk(folder):
        for f in fs:
            p = os.path.join(d, f)
            h = hashlib.sha256()
            with open(p, 'rb') as fh:
                for block in iter(lambda: fh.read(1 << 24), b''):
                    h.update(block)
            out[os.path.relpath(p, folder)] = h.hexdigest()
    return out


def prune(name):
    pod_out = f'/workspace/runs/{name}/mpd-data/runpod/{name}'
    for sub in ('run', 'warm'):
        listing = on_pod(name, f'ls {pod_out}/{sub} 2> /dev/null || true').split()
        steps = sorted((int(m.group(1)) for n in listing for m in [re.match(r'step_(\d+)$', n)] if m))
        for s in steps[:-KEEP]:
            rel = f'{sub}/step_{s}'
            mac = os.path.join(RUNPOD, name, rel)
            if not os.path.isdir(mac):
                print(f'{name} {rel}: not on the Mac yet', flush=True)
                continue
            pod = {}
            for line in on_pod(name, f'cd {pod_out}/{rel} && find . -type f -exec sha256sum {{}} +').splitlines():
                h, p = line.split(None, 1)
                pod[os.path.normpath(p)] = h
            if pod and pod == mac_hashes(mac):
                on_pod(name, f'rm -rf {pod_out}/{rel}')
                print(f'{time.strftime("%H:%M")} {name} {rel}: removed from the pod (Mac copy hash-matches)', flush=True)
            else:
                print(f'{name} {rel}: Mac copy differs or is incomplete; left on the pod', flush=True)


if __name__ == '__main__':
    names = sys.argv[1:]
    while names:
        for n in list(names):
            if not os.path.exists(os.path.join(RUNPOD, n, 'pod.txt')) or 'ended' in open(os.path.join(RUNPOD, n, 'rp-run.log')).read().split('\n')[-2]:
                names.remove(n)
                continue
            try:
                prune(n)
            except Exception as e:
                print(f'{n}: {e}', flush=True)
        time.sleep(600)
