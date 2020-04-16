#!/bin/bash
check_n_state() {
  echo "1000 1000 $1" > gen.in
  ./bin/generator < gen.in > tmp.in
  ./bin/driver_cuda < tmp.in >> n_obs_1000_n_states_1000.out

}

for ((i = 100; i < 1000; i += 20)); do
  check_n_state $i
  echo "1000 1000 $i done."
done
rm -rf "*.in"