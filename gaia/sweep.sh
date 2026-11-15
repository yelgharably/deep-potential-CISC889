run(){ python gaia_phi.py "$@" > /dev/null 2>&1; }
for s in 1 2; do run --arm baseline --seed $s & run --arm monopos --seed $s & run --arm lower --seed $s --w 1 & done; wait
for s in 0 1 2; do run --arm lower --seed $s --w 0.1 & run --arm lower --seed $s --w 0.3 & done; wait
