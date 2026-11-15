run(){ python gaia_phi.py "$@" > /dev/null 2>&1; }
for s in 0 1 2; do run --arm lohi --hi-scale 1.3 --pool-rmax 0.6 --n-pool 600 --seed $s & run --arm lohi --hi-scale 1.6 --pool-rmax 0.6 --n-pool 600 --seed $s & run --arm lower --w 0.3 --pool-rmax 0.6 --n-pool 600 --seed $s & done; wait
