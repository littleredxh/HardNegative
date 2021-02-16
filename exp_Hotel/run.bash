DATA="HOTEL"
MODEL="R50"
DIM=256
EP=200
NGPU=4
REPEAT=5
BSIZE=512
LAM=1
for METHOD in "shn"; do
for LR in "1e-2" "3e-2" "1e-1" "3e-1"; do
for IT in 1 2; do
#
echo "method: ${METHOD} lam: ${LAM}, lr: ${LR}, i: ${IT}"
export MODEL DATA DIM LR EP IT BSIZE NGPU REPEAT BSIZE LAM METHOD
#
sbatch Hotel_L.sbatch
#
sleep .1 # pause to be kind to the scheduler
done
done
done
#