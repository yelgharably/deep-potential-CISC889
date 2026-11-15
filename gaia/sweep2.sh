run(){ python gaia_phi.py "$@" > /dev/null 2>&1; }
for s in 0 1 2; do run --arm lohi --hi-scale 1.3 --seed $s & run --arm lohi --hi-scale 1.6 --seed $s & done; wait
