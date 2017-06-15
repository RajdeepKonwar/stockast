#! /bin/bash

> time_local.log
for ((i = 1; i <=8; i++));
do
  echo $i;
  export OMP_NUM_THREADS=$i;
  time ./stockast 0 >> time_local.log;
done
