MAKEFLAGS+= -j9
NVCC=		nvcc
NVCC_OPTIM_FLAGS= --device-c -arch=sm_70 -rdc=true -lcudadevrt --ptxas-options=-v
NVCC_DEBUG_FLAGS= -g -G -O0 --device-c -arch=sm_70
BINDIR= 	bin
SOURCEDIR=	source
TESTDIR=	unitTests

PROG=memoryInitTest unitTest1 unitTest2
BINLIST=$(addprefix $(BINDIR)/, $(PROG))

ifeq ($(debug), 1)
	NVCC_FLAGS = $(NVCC_DEBUG_FLAGS)
	NVCC_LINK_FLAG = -arch=sm_70 -O0 -g -G -lcudadevrt --ptxas-options=-v
else
	NVCC_FLAGS = $(NVCC_OPTIM_FLAGS)
	NVCC_LINK_FLAG = -arch=sm_70 -O4 -lcudadevrt --ptxas-options=-v
endif

all: $(BINDIR) $(BINLIST)

$(BINDIR):
	mkdir $(BINDIR)

$(BINDIR)/%:  $(TESTDIR)/%.cu $(SOURCEDIR)/Paging.cuh
	$(NVCC) $< -o $@ $(NVCC_LINK_FLAG)


parallelPage.o: parallelPage.cuh parallelPage.cu
	nvcc $(NVCC_FLAGS) parallelPage.cu -o parallelPage.o

memoryInitTest: unitTests/memoryInitTest.cu source/Paging.cuh
	nvcc $(NVCC_LINK_FLAG) unitTests/memoryInitTest.cu -o memoryInitTest

unitTestRandomWalk.o: parallelPage.o unitTestRandomWalk.cu metrics.h
	nvcc $(NVCC_FLAGS) unitTestRandomWalk.cu -o unitTestRandomWalk.o

unitTestRandomWalk: unitTestRandomWalk.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestRandomWalk.o -o unitTestRandomWalk

unitTestRandomWalk2.o: unitTestRandomWalk2.cu parallelPage.o
	nvcc $(NVCC_FLAGS) unitTestRandomWalk2.cu -o unitTestRandomWalk2.o

unitTestRandomWalk2: unitTestRandomWalk2.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestRandomWalk2.o -o unitTestRandomWalk2

unitTestClusteredRandomWalk2.o: unitTestClusteredRandomWalk2.cu parallelPage.o
	nvcc $(NVCC_FLAGS) unitTestClusteredRandomWalk2.cu -o unitTestClusteredRandomWalk2.o

unitTestClusteredRandomWalk2: unitTestClusteredRandomWalk2.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestClusteredRandomWalk2.o -o unitTestClusteredRandomWalk2

unitTestClusteredRandomWalk.o: parallelPage.o unitTestClusteredRandomWalk.cu metrics.h
	nvcc $(NVCC_FLAGS) unitTestClusteredRandomWalk.cu -o unitTestClusteredRandomWalk.o

unitTestClusteredRandomWalk: unitTestClusteredRandomWalk.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestClusteredRandomWalk.o -o unitTestClusteredRandomWalk

unitTestLinkedList.o: parallelPage.o unitTestLinkedList.cu metrics.h
	nvcc $(NVCC_FLAGS) unitTestLinkedList.cu -o unitTestLinkedList.o

unitTestLinkedList: unitTestLinkedList.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestLinkedList.o -o unitTestLinkedList

unitTestLinkedList0.o: parallelPage.o unitTestLinkedList0.cu metrics.h
	nvcc $(NVCC_FLAGS) unitTestLinkedList0.cu -o unitTestLinkedList0.o

unitTestLinkedList0: unitTestLinkedList0.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestLinkedList0.o -o unitTestLinkedList0

# gather clock cycles information
unitTestSingleClock0.o: parallelPage.o unitTestSingleClock0.cu
	nvcc $(NVCC_FLAGS) unitTestSingleClock0.cu -o unitTestSingleClock0.o

unitTestSingleClock0: unitTestSingleClock0.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestSingleClock0.o -o unitTestSingleClock0

# unit test for number of steps and time
unitTestSingleClock1.o: parallelPage.o unitTestSingleClock1.cu
	nvcc $(NVCC_FLAGS) unitTestSingleClock1.cu -o unitTestSingleClock1.o

unitTestSingleClock1: unitTestSingleClock1.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestSingleClock1.o -o unitTestSingleClock1

# gather clock cycles information
unitTestDistClock0.o: parallelPage.o unitTestDistClock0.cu
	nvcc $(NVCC_FLAGS) unitTestDistClock0.cu -o unitTestDistClock0.o

unitTestDistClock0: unitTestDistClock0.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestDistClock0.o -o unitTestDistClock0

# unit test for number of steps and time
unitTestDistClock1.o: parallelPage.o unitTestDistClock1.cu
	nvcc $(NVCC_FLAGS) unitTestDistClock1.cu -o unitTestDistClock1.o

unitTestDistClock1: unitTestDistClock1.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestDistClock1.o -o unitTestDistClock1

unitTestCollabRW.o: parallelPage.o unitTestCollabRW.cu
	nvcc $(NVCC_FLAGS) unitTestCollabRW.cu -o unitTestCollabRW.o

unitTestCollabRW: unitTestCollabRW.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestCollabRW.o -o unitTestCollabRW

unitTestCollabRW_BM.o: parallelPage.o unitTestCollabRW_BM.cu
	nvcc $(NVCC_FLAGS) unitTestCollabRW_BM.cu -o unitTestCollabRW_BM.o 

unitTestCollabRW_BM: unitTestCollabRW_BM.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestCollabRW_BM.o -o unitTestCollabRW_BM

unitTestRandomWalk_BM.o: parallelPage.o unitTestRandomWalk_BM.cu
	nvcc $(NVCC_FLAGS) unitTestRandomWalk_BM.cu -o unitTestRandomWalk_BM.o

unitTestRandomWalk_BM: unitTestRandomWalk_BM.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestRandomWalk_BM.o -o unitTestRandomWalk_BM

unitTestClusteredRandomWalk_BM.o: parallelPage.o unitTestClusteredRandomWalk_BM.cu
	nvcc $(NVCC_FLAGS) unitTestClusteredRandomWalk_BM.cu -o unitTestClusteredRandomWalk_BM.o

unitTestClusteredRandomWalk_BM: unitTestClusteredRandomWalk_BM.o parallelPage.o
	nvcc -arch=sm_70 parallelPage.o unitTestClusteredRandomWalk_BM.o -o unitTestClusteredRandomWalk_BM

malloc.o: parallelPage.o malloc.cuh malloc.cu
	nvcc $(NVCC_FLAGS) -rdc=true -lcudadevrt malloc.cu -o malloc.o	

unitTestMalloc1.o: unitTestMalloc1.cu
	nvcc $(NVCC_FLAGS) -rdc=true -lcudadevrt unitTestMalloc1.cu -o unitTestMalloc1.o 

unitTestMalloc1: parallelPage.o malloc.o unitTestMalloc1.o
	nvcc -arch=sm_70 parallelPage.o malloc.o unitTestMalloc1.o -o unitTestMalloc1

unitTestMalloc2.o: unitTestMalloc2.cu
	nvcc $(NVCC_FLAGS) -rdc=true -lcudadevrt unitTestMalloc2.cu -o unitTestMalloc2.o 

unitTestMalloc2: parallelPage.o malloc.o unitTestMalloc2.o
	nvcc -arch=sm_70 parallelPage.o malloc.o unitTestMalloc2.o -o unitTestMalloc2


# unitTestRandomWalk: libParallelPage.cuh unitTestRandomWalk.cu
# 	nvcc unitTestRandomWalk.cu -o unitTestRandomWalk

# unitTestClusteredRandomWalk: libParallelPage.cuh unitTestClusteredRandomWalk.cu
# 	nvcc unitTestClusteredRandomWalk.cu -o unitTestClusteredRandomWalk

# unitTestLinkedList: libParallelPage.cuh unitTestLinkedList.cu
# 	nvcc unitTestLinkedList.cu -o unitTestLinkedList

.PHONY: test debug clean
clean:
	rm -f bin/*
debug:
	make clean && make -j10 debug=1 && CUDA_VISIBLE_DEVICES=0 cuda-gdb bin/unitTest2
