This program has been modified to evaluate the performance of several random
number libraries on GPU. Each port is in their own branch. The libraries, along with 
the branch names, are:

+ cuRAND (cuda)
+ rocRAND (roc)
+ Random123 (r123)
+ Kokkos's random module (kokkos)
+ RND (rng)


## Usage
To checkout one of the implementations, say RND, use the following command:

```
git checkout rng
mkdir build; cd build
cmake ..
make -j4
```

This will generate an output called `rt` in `build/` directory. To run the program, use the following command:

```
./rt > out.ppm
```

This will generate a ppm file called `out.ppm` which can be viewed using any image viewer. 

For cuda and rocRAND, cmake configuration step is not needed.

## Results
The following results are for Nvidia A100 80GB SMX and V100 32GB GPUs. The program renders a 1200 X 800 image 
with 50 samples per pixel.

| Library | V100 (secs)| A(100) |
|---------|------------|--------|
| cuRAND  | 4.03       | 2.91 |
| rocRAND | X    | 2.93 |
| Random123| 3.88    | 2.84 |
| Kokkos  | 6.02      | 2.70 |
| RND     | 4.05      | 2.78 |






