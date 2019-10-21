SRC=src
BIN=bin
UTIL=util
DRIVER=driver.cu
VITERBI_CUDA=viterbi_cuda.cu
VITERBI_SEQUENTIAL=viterbi_sequential.cu
GENERATOR=generator.c

echo -e "[*] use nvcc to compile \n\t{$DRIVER, $VITERBI_CUDA,$VITERBI_SEQUENTIAL}\n\t... \c"
nvcc -rdc=true \
  -arch=sm_60 \
  $SRC/$DRIVER \
  $SRC/$VITERBI_SEQUENTIAL \
  $SRC/$VITERBI_CUDA \
  -o $BIN/driver_cuda \
  -D DEBUG
echo -e "done."
echo -e "\t--> bin/driver_cuda"

echo -e "[*] use gcc to compile ... \c"
gcc $UTIL/$GENERATOR -o $BIN/generator
echo -e "done."
echo -e "\t--> bin/generator"
