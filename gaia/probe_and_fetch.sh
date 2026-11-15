#!/bin/bash
# wait until the archive answers a small async query, then run the resumable fetch (cached tiles are skipped)
until timeout 90 python tiletest.py 2>/dev/null | grep -q "COMPLETED"; do echo "$(date) archive still unresponsive"; sleep 180; done
echo "$(date) archive back; fetching"
python gaia_fetch.py --dmax 1.0 --limit 400000 --out gaia_6d.npz
