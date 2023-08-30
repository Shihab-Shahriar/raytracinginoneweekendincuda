# Do this before running the executable:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/ufs18/home-158/khanmd/hippy/clr/build/install/lib

HOST_COMPILER  = g++
NVCC           = nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) 
GENCODE_FLAGS  = -arch=sm_80 
#-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75

SRCS = main.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h
INCLUDE_PATH = -I/mnt/ufs18/home-158/khanmd/hippy/clr/build/install/include
LIBS = -L/mnt/ufs18/home-158/khanmd/hippy/clr/build/install/lib
LDFLAGS = -lrocrand

cudart: cudart.o $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o $(LIBS) $(LDFLAGS)

cudart.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDE_PATH) -o cudart.o -c main.cu 


out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: cudart
	nvprof ./cudart > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

clean:
	rm -f cudart cudart.o out.ppm out.jpg out.txt
