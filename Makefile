NVCC=		nvcc
NVCC_OPTIM_FLAGS= -Xptxas -O4 -Xcompiler -O4 --device-c -arch=sm_70
NVCC_DEBUG_FLAGS= -g -G -O0 --device-c -arch=sm_70
ifeq ($(debug), 1)
	NVCC_FLAGS = $(NVCC_DEBUG_FLAGS)
else
	NVCC_FLAGS = $(NVCC_OPTIM_FLAGS)
endif

all: unitTestRandomWalk unitTestClusteredRandomWalk unitTestLinkedList

parallelPage.o: parallelPage.cuh parallelPage.cu
	nvcc $(NVCC_FLAGS) parallelPage.cu -o parallelPage.o

unitTestRandomWalk.o: parallelPage.o unitTestRandomWalk.cu metrics.h
	nvcc $(NVCC_FLAGS) unitTestRandomWalk.cu -o unitTestRandomWalk.o

unitTestRandomWalk: unitTestRandomWalk.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestRandomWalk.o -o unitTestRandomWalk

unitTestClusteredRandomWalk.o: parallelPage.o unitTestClusteredRandomWalk.cu metrics.h
	nvcc $(NVCC_FLAGS) unitTestClusteredRandomWalk.cu -o unitTestClusteredRandomWalk.o

unitTestClusteredRandomWalk: unitTestClusteredRandomWalk.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestClusteredRandomWalk.o -o unitTestClusteredRandomWalk

unitTestLinkedList.o: parallelPage.o unitTestLinkedList.cu metrics.h
	nvcc $(NVCC_FLAGS) unitTestLinkedList.cu -o unitTestLinkedList.o

unitTestLinkedList: unitTestLinkedList.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestLinkedList.o -o unitTestLinkedList

# unitTestRandomWalk: libParallelPage.cuh unitTestRandomWalk.cu
# 	nvcc unitTestRandomWalk.cu -o unitTestRandomWalk

# unitTestClusteredRandomWalk: libParallelPage.cuh unitTestClusteredRandomWalk.cu
# 	nvcc unitTestClusteredRandomWalk.cu -o unitTestClusteredRandomWalk

# unitTestLinkedList: libParallelPage.cuh unitTestLinkedList.cu
# 	nvcc unitTestLinkedList.cu -o unitTestLinkedList

.PHONY: test debug clean
clean:
	rm -f *.o *.a unitTestRandomWalk unitTestClusteredRandomWalk unitTestLinkedList
debug:
	make clean && make -j10 debug=1 && CUDA_VISIBLE_DEVICES=0 cuda-gdb unitTestLinkedList
