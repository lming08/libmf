#!/bin/sh

set -e

./libmf convert ../ua.base smalldata.tr.bin
./libmf convert ../ua.test smalldata.te.bin
./libmf train --tr-rmse --obj -k 100 -t 70 -s 4 -p 0.008 -q 0.012 -g 0.0025 -ub -1 -ib -1 --no-use-avg --rand-shuffle -v smalldata.te.bin smalldata.tr.bin model
./libmf predict smalldata.te.bin model output
./libmf similarity smalldata.te.bin model euclidean
