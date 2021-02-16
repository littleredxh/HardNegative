DATA="ICR"
MODEL="R50"
DIM=512
EP=200
NGPU=4
REPEAT=5
BSIZE=512
LAM=1
for METHOD in "sct" "shn"; do
for LR in  "3e-2" "1e-1" "3e-1" "1e-0" "3e-0"; do
for IT in 0 1 2 3 4; do
#
echo "lam: ${LAM}, lr: ${LR}, i: ${IT}"
export MODEL DATA DIM LR EP IT BSIZE NGPU REPEAT BSIZE LAM METHOD
#
sbatch Inshop_L.sbatch
#
sleep .1 # pause to be kind to the scheduler
done
done
done
#