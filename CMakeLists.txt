cmake_minimum_required(VERSION 3.6)
project(mpcp2)

# https://stackoverflow.com/questions/23163075/how-to-compile-an-mpi-included-c-program-using-cmake
# set(INC_DIR /home/q/.openmpi/include)
# set(LINK_DIR /home/q/.openmpi/lib/)

# include_directories(${INC_DIR})
# link_directori/home/qes(${LINK_DIR})
# link_libraries(MPI_4)

#set(SOURCE_FILES util/generator.c src/mpi/knapsack.c src/orig/knapsack.c)
#set(SOURCE_FILES src/viterbi_sequential.c)
set(SOURCE_FILES src/viterbi_sequential.cu src/driver.cu headers/driver.h
        headers/viterbi_sequential.h)
#set(SOURCE_FILES src/viterbi_sequential.c util/generator.c headers/driver.h
#        headers/viterbi_sequential.h)
# set(LD_LIBRARY_PATH /home/q/.openmpi/lib/ )
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
 target_link_libraries(${PROJECT_NAME} m)
# target_link_libraries(${PROJECT_NAME} MPI_4)