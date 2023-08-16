<<<<<<< HEAD
CXX = g++

rt: main.cc camera.h hitable.h hitable_list.h material.h ray.h sphere.h vec3.h
	$(CXX) -O3 -o rt main.cc

out.ppm: rt
	rm -f out.ppm
	time ./rt > out.ppm
=======
CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c main.cu

out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm
>>>>>>> original

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

<<<<<<< HEAD
clean:
	rm -f rt rt.o out.ppm out.jpg
=======
profile_basic: cudart
	nvprof ./cudart > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

clean:
	rm -f cudart cudart.o out.ppm out.jpg
>>>>>>> original
