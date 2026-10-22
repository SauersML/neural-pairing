#!/bin/bash
# Sends a checkpoint from a training pod straight to the evaluation worker's pod (evalworker.sh), pod to pod:
# send_job.sh SRC_RUN STEP TAG [WORKER_RUN]. The training pod gets its own key (made once there, never leaves it), the
# worker accepts it, and the step folder, the run's replies.json and data/ go across in seconds; READY is written last.
set -eu
SRC=$1 STEP=$2 TAG=$3 W=${4:-magic-nl2w}
P=/Users/user/mpd-data/neural-pairing/podcp.sh
WPOD=$(tail -1 /Users/user/mpd-data/runpod/$W/pod.txt | cut -d" " -f1)
read -r WIP WPORT _ _ < /Users/user/mpd-data/runpod/active/.$WPOD.ssh
PUB=$(bash $P $SRC --run "[ -f /root/.ssh/p2p ] || ssh-keygen -t ed25519 -N '' -q -f /root/.ssh/p2p; cat /root/.ssh/p2p.pub" 2> /dev/null)
bash $P $W --run "mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys && (grep -qF '$PUB' /root/.ssh/authorized_keys || echo '$PUB' >> /root/.ssh/authorized_keys)" 2> /dev/null
O=/workspace/runs/$SRC/mpd-data/runpod/$SRC
SSH="ssh -i /root/.ssh/p2p -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $WPORT root@$WIP"
bash $P $SRC --run "set -e; [ -d $O/run/step_$STEP ]; tar -C $O/run -cf - step_$STEP -C $O replies.json -C /workspace/runs/$SRC/src/w data | $SSH 'mkdir -p /workspace/jobs/$TAG && tar -C /workspace/jobs/$TAG -xf - && touch /workspace/jobs/$TAG/READY && du -sh /workspace/jobs/$TAG'" 2> /dev/null
echo "$(date '+%H:%M:%S') sent $SRC step_$STEP as $TAG to $W"
