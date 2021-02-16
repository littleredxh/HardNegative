DATA="CAR"
MODEL="R50"
DIM=64
EP=80
NGPU=1
REPEAT=1
BSIZE=128
LAM=1
for METHOD in "sct"; do
for LR in "2e-3"; do
for IT in 0; do
#
echo "method: ${METHOD} lam: ${LAM}, lr: ${LR}, i: ${IT}"
export MODEL DATA DIM LR EP IT BSIZE NGPU REPEAT BSIZE LAM METHOD
#
sbatch CAR_D.sbatch
#
sleep .1 # pause to be kind to the scheduler
done
done
done
#"shn" "1e-3" "2e-3" "3e-3" "4e-3"
#"sct" "1e-2" "2e-2" "3e-2" "4e-2"