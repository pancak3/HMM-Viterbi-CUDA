echo -e "[*] use nvcc to compile ... \c"
nvcc -rdc=true \
 src/driver.cu \
 src/viterbi_sequential.cu \
 src/viterbi_cuda.cu \
 -o bin/driver_cuda \
 -D DEBUG
echo -e "done."
echo -e "[*] --> bin/driver_cuda"

