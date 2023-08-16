Ray Tracing in One Weekend in CUDA
==================================

<<<<<<< HEAD
This is yet another _Ray Tracing in One Weekend_ clone, but this time using CUDA instead of C++.  CUDA can be used to speed up the code.  For example, on my machine, the C++ code renders the test image in 90 seconds.  The CUDA accelerated code renders the image in about 7 seconds.

Initial coding started in May, 2018 and was posted to the NVIDIA Developer blog November 5, 2018: https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/

Background
----------

Peter Shirley has written a few ebooks about Ray Tracing.  You can find out more at http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html  Note that as of April, 2018 the books are *pay what you wish* and 50% of the proceeds go towards not-for-profit programming education organizations.  They are also available for $3 each on Amazon as a Kindle download.

This repository contains code for converting the first ray tracer ebook "Ray Tracing in one Weekend" from C++ to CUDA.  By changing to CUDA, depending on your CPU and GPU you can see speedups of 10x or more!  _UPDATE: see [Issue #2](https://github.com/rogerallen/raytracinginoneweekendincuda/issues/2) for a further 2x improvement!_

Before coding the ray tracer in CUDA, I recommend that you code the ray tracer in C++, first.  You should understand the concepts presented in a serial language well, then translate this knowledge to CUDA.  In fact, since CUDA uses C++, much of your code can be reused.

The C++ code that this repository is based on is at https://github.com/petershirley/raytracinginoneweekend.  As of January, 2020, the book and code are being updated and improved at https://github.com/RayTracing/raytracing.github.io/.  This repository has not been changed to match these changes.  The code matches the original book from 2016 which you can still download from http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html.  Further, I am basing this on the repo at https://github.com/pfranz/raytracinginoneweekend which has each chapter as a separate git branch.  This is very handy for checking out the code at each chapter.

Chapter List
------------

Here are links to the git branch for each Chapter.  If you look at the README.md you'll see some hints about what needed to be done.  See the Makefile for the standard targets.  Note that you'll want to adjust the GENCODE_FLAGS in the CUDA Makefiles for your specific graphics card architecture.

The master branch has the code as Peter Shirley presented it in C++.  I added a Makefile so you can `make out.jpg` and compare the runtime to CUDA.  To build variants that use CUDA, check out one of these branches.  E.g. `git checkout ch01_output_cuda`

* [Chapter 1 - Basic Output](https://github.com/rogerallen/raytracinginoneweekend/tree/ch01_output_cuda): `git checkout ch01_output_cuda`
* [Chapter 2 - Vectors](https://github.com/rogerallen/raytracinginoneweekend/tree/ch02_vec3_cuda): `git checkout ch02_vec3_cuda`
* [Chapter 3 - Rays](https://github.com/rogerallen/raytracinginoneweekend/tree/ch03_rays_cuda): `git checkout ch03_rays_cuda`
* [Chapter 4 - Spheres](https://github.com/rogerallen/raytracinginoneweekend/tree/ch04_sphere_cuda): `git checkout ch04_sphere_cuda`
* [Chapter 5 - Normals](https://github.com/rogerallen/raytracinginoneweekend/tree/ch05_normals_cuda): `git checkout ch05_normals_cuda`
* [Chapter 6 - Antialiasing](https://github.com/rogerallen/raytracinginoneweekend/tree/ch06_antialiasing_cuda): `git checkout ch06_antialiasing_cuda`
* [Chapter 7 - Diffuse](https://github.com/rogerallen/raytracinginoneweekend/tree/ch07_diffuse_cuda): `git checkout ch07_diffuse_cuda`
* [Chapter 8 - Metal](https://github.com/rogerallen/raytracinginoneweekend/tree/ch08_metal_cuda): `git checkout ch08_metal_cuda`
* [Chapter 9 - Dielectrics](https://github.com/rogerallen/raytracinginoneweekend/tree/ch09_dielectrics_cuda): `git checkout ch09_dielectrics_cuda`
* [Chapter 10 - Camera](https://github.com/rogerallen/raytracinginoneweekend/tree/ch10_camera_cuda): `git checkout ch10_camera_cuda`
* [Chapter 11 - Defocus Blur](https://github.com/rogerallen/raytracinginoneweekend/tree/ch11_defocus_blur_cuda): `git checkout ch11_defocus_blur_cuda`
* [Chapter 12 - Where Next](https://github.com/rogerallen/raytracinginoneweekend/tree/ch12_where_next_cuda): `git checkout ch12_where_next_cuda`

Colophon
--------

Basic process (after Chapter 3) was:

```
# checkout original code & create a cuda branch
git checkout origin/chyy_yyy
git checkout chyy_yyy
git branch -m chyy_yyy_cuda
git mv main.cc main.cu
<checkin>
# grab previous chapters code as a starting point
cp chapterxx/* .
# edit & fix code
# checkin code
# save current code for next chapter
mkdir chapteryy
cp * chapteryy
```
=======
This is yet another _Ray Tracing in One Weekend_ clone, but this time using CUDA instead of C++.

By Roger Allen
May, 2018

See the [Master Branch](https://github.com/rogerallen/raytracinginoneweekend) for more information.

Chapter 1
---------

This introduces the basic kernel launch mechanism & host/device memory management.  We are just creating an image on the GPU device and cudaMallocmanaged allows for sharing the framebuffer and automatically copying that buffer to & from the device.

I also added a timer to see how long it takes the GPU to do rendering.

Chapter 2
---------

Because CUDA is compatible with C++ and the vec3.h class will be used on both GPU & CPU, we add `__host__` `__device__` as a prefix to all methods.

Chapter 3
---------

Since the ray class is only used on the GPU, we will just add `__device__` as a prefix to all methods.

The color function just needs a `__device__` added since this is called from the render kernel.

Note, doing a straight translation from the original C++ will mean that any floating-point constants will be doubles and math on the GPU will be forced to be double-precision.  This will hurt our performance unnecessarily.  Special attention to floating point constants must be taken (e.g. 0.5 -> 0.5f).

Use the "profile_metrics" makefile target to count inst_fp_64 and be sure that is 0.

Chapter 4
---------

We only need to add a `__device__` to the hit_sphere() call and use profile_metrics to watch for those floating-point constants.

Chapter 5
---------

Here we have to create our world of spheres on the device and get familiar with how we do memory management for CUDA C++ classes.  Note the cudaMalloc of `d_list` and `d_world` and the `create_world` kernel.

Again, attend to `__device__` and floating-point constants in hitable.h, hitable_list.h and sphere.h.

Chapter 6
---------

In this chapter we need to understand using cuRAND for per-thread random numbers.  See `d_rand_state` and `render_init`.

Note that now using debug flags in compilation makes a big difference in runtime.  Remove those flags for a signficant speedup.

Chapter 7
---------

Matching the C++ code in the color function in main.cu would recurse enough into the color() calls that it was crashing the program by overrunning the stack, so we turn this function into a limited-depth loop instead.  Later code in the book limits to a max depth of 50, so we adapt this a few chapters early on the GPU.

Chapter 8
---------

Just more plumbing for per-thread local random state, mostly.

Chapter 9
---------

Similar to previous modifications.

Chapter 10
----------

Similar to previous modifications.

Chapter 11
----------

Similar to previous modifications.

Chapter 12
----------

And we're done!
>>>>>>> original
