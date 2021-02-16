DATA="CUB"
MODEL="R50"
DIM=64
EP=80
NGPU=1
REPEAT=1
BSIZE=128
for LAM in 1; do
for METHOD in "sct"; do
for LR in "9e-3"; do
for IT in 0; do
#
echo "method: ${METHOD} lam: ${LAM}, lr: ${LR}, i: ${IT}"
export MODEL DATA DIM LR EP IT BSIZE NGPU REPEAT BSIZE LAM METHOD
#
sbatch CUB_D.sbatch
#
sleep .1 # pause to be kind to the scheduler
done
done
done
done
# "sct" "6e-3" "9e-3" "12e-3"
# "shn" "1e-3" "2e-3" "3e-3"