#!/bin/bash
# Lead-only pod-to-pod handoff, matching send_nl22.sh. No files transit the Mac.
# Source must still have nl22c's step_157 and w/data. Override SRC_DATA if that pod used another work directory.
set -euo pipefail
P=/Users/user/mpd-data/neural-pairing/podcp.sh
SRC=${1:-magic-nl22c} W=${2:-magic-nl24}
O=/workspace/runs/$SRC/mpd-data/runpod/$SRC
D=${SRC_DATA:-/workspace/runs/$SRC/w}
until [ -s "/Users/user/mpd-data/runpod/$W/pod.txt" ] && bash "$P" "$W" --run 'mkdir -p /workspace/jobs'; do sleep 20; done
WPOD=$(tail -1 "/Users/user/mpd-data/runpod/$W/pod.txt" | cut -d' ' -f1)
read -r WIP WPORT _ _ < "/Users/user/mpd-data/runpod/active/.$WPOD.ssh"
PUB=$(bash "$P" "$SRC" --run "test -s '$O/run/step_157/bridge.pt' && test -s '$D/data/chat.json' && { [ -f /root/.ssh/p2p ] || ssh-keygen -t ed25519 -N '' -q -f /root/.ssh/p2p; } && cat /root/.ssh/p2p.pub")
bash "$P" "$W" --run "mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys && (grep -qF '$PUB' /root/.ssh/authorized_keys || echo '$PUB' >> /root/.ssh/authorized_keys)"
SSH="ssh -i /root/.ssh/p2p -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $WPORT root@$WIP"
bash "$P" "$SRC" --run "set -eo pipefail; tar -C '$O/run' -cf - step_157 -C '$O' replies.json -C '$D' data | $SSH 'mkdir -p /workspace/jobs/nl24 && tar -C /workspace/jobs/nl24 -xf - && test -s /workspace/jobs/nl24/step_157/bridge.pt && test -s /workspace/jobs/nl24/data/chat.json && touch /workspace/jobs/nl24/READY && du -sh /workspace/jobs/nl24'"
echo "sent step_157 and nl22 data from $SRC to $W"
