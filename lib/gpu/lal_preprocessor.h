// **************************************************************************
//                              preprocessor.cu
//                             -------------------
//                           W. Michael Brown (ORNL)
//
//  Device code for CUDA-specific preprocessor definitions
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                : 
//    email                : brownw@ornl.gov
// ***************************************************************************/

//*************************************************************************
//                           Preprocessor Definitions
//                           
//  Note: It is assumed that constants with the same names are defined with
//  the same values in all files.
//  
//  ARCH
//     Definition:   Architecture number for accelerator
//  MEM_THREADS
//     Definition:   Number of threads with sequential ids accessing memory
//                   simultaneously on multiprocessor
//  WARP_SIZE:
//     Definition:   Number of threads guaranteed to be on the same instruction
//  THREADS_PER_ATOM
//     Definition:   Default number of threads assigned per atom for pair styles
//     Restructions: Must be power of 2; THREADS_PER_ATOM<=WARP_SIZE
//  THREADS_PER_CHARGE
//     Definition:   Default number of threads assigned per atom for pair styles
//                   with charge
//     Restructions: Must be power of 2; THREADS_PER_ATOM<=WARP_SIZE
//  PPPM_MAX_SPLINE
//     Definition:   Maximum order for splines in PPPM
//  PPPM_BLOCK_1D    
//     Definition:   Thread block size for PPPM kernels
//     Restrictions: PPPM_BLOCK_1D>=PPPM_MAX_SPLINE*PPPM_MAX_SPLINE
//                   PPPM_BLOCK_1D%32==0 
//  BLOCK_PAIR
//     Definition:   Default thread block size for pair styles
//     Restrictions:
//  MAX_SHARED_TYPES 8
//     Definition:   Max # of atom type params can be stored in shared memory
//     Restrictions: MAX_SHARED_TYPES*MAX_SHARED_TYPES<=BLOCK_PAIR
//  BLOCK_CELL_2D 
//     Definition:   Default block size in each dimension for cell list builds
//                   and matrix transpose
//  BLOCK_CELL_ID    
//     Definition:   Default block size for binning atoms in cell list builds
//  BLOCK_NBOR_BUILD 
//     Definition:   Default block size for neighbor list builds
//  BLOCK_BIO_PAIR
//     Definition:   Default thread block size for "bio" pair styles
//  MAX_BIO_SHARED_TYPES
//     Definition:   Max # of atom type params can be stored in shared memory
//     Restrictions:  MAX_BIO_SHARED_TYPES<=BLOCK_BIO_PAIR*2 &&
//                    MAX_BIO_SHARED_TYPES>=BLOCK_BIO_PAIR
//
//*************************************************************************/

// -------------------------------------------------------------------------
//                            CUDA DEFINITIONS
// -------------------------------------------------------------------------

#ifdef NV_KERNEL

#define GLOBAL_ID_X threadIdx.x+mul24(blockIdx.x,blockDim.x)
#define GLOBAL_ID_Y threadIdx.y+mul24(blockIdx.y,blockDim.y)
#define GLOBAL_SIZE_X mul24(gridDim.x,blockDim.x);
#define GLOBAL_SIZE_Y mul24(gridDim.y,blockDim.y);
#define THREAD_ID_X threadIdx.x
#define THREAD_ID_Y threadIdx.y
#define BLOCK_ID_X blockIdx.x
#define BLOCK_ID_Y blockIdx.y
#define BLOCK_SIZE_X blockDim.x
#define BLOCK_SIZE_Y blockDim.y
#define __kernel extern "C" __global__
#define __local __shared__
#define __global  
#define atom_add atomicAdd
#define ucl_inline static __inline__ __device__ 

#ifdef __CUDA_ARCH__
#define ARCH __CUDA_ARCH__
#else
#define ARCH 100
#endif

#if (ARCH < 200)

#define THREADS_PER_ATOM 1
#define THREADS_PER_CHARGE 16
#define BLOCK_NBOR_BUILD 64
#define BLOCK_PAIR 64
#define BLOCK_BIO_PAIR 64
#define MAX_SHARED_TYPES 8

#else

#if (ARCH < 300)

#define THREADS_PER_ATOM 4
#define THREADS_PER_CHARGE 8
#define BLOCK_NBOR_BUILD 128
#define BLOCK_PAIR 128
#define BLOCK_BIO_PAIR 128
#define MAX_SHARED_TYPES 11

#else

#define THREADS_PER_ATOM 4
#define THREADS_PER_CHARGE 8
#define BLOCK_NBOR_BUILD 128
#define BLOCK_PAIR 512
#define BLOCK_BIO_PAIR 512
#define MAX_SHARED_TYPES 11

#endif

#endif

#define WARP_SIZE 32
#define PPPM_BLOCK_1D 64
#define BLOCK_CELL_2D 8
#define BLOCK_CELL_ID 128
#define MAX_BIO_SHARED_TYPES 128

#ifdef _DOUBLE_DOUBLE
ucl_inline double4 fetch_pos(const int& i, const double4 *pos) { return pos[i]; };
ucl_inline double fetch_q(const int& i, const double *q) { return q[i]; };
#endif

#if (__CUDA_ARCH__ < 200)
#define fast_mul __mul24
#define MEM_THREADS 16
#else
#define fast_mul(X,Y) (X)*(Y)
#define MEM_THREADS 32
#endif

#ifdef CUDA_PRE_THREE
struct __builtin_align__(16) _double4
{
  double x, y, z, w;
};
typedef struct _double4 double4;
#endif

#ifdef _DOUBLE_DOUBLE

#define ucl_exp exp
#define ucl_powr pow
#define ucl_atan atan
#define ucl_cbrt cbrt
#define ucl_ceil ceil
#define ucl_abs fabs
#define ucl_rsqrt rsqrt
#define ucl_sqrt sqrt
#define ucl_recip(x) ((numtyp)1.0/(x))

#else

#define ucl_atan atanf
#define ucl_cbrt cbrtf
#define ucl_ceil ceilf
#define ucl_abs fabsf
#define ucl_recip(x) ((numtyp)1.0/(x))
#define ucl_rsqrt rsqrtf
#define ucl_sqrt sqrtf

#ifdef NO_HARDWARE_TRANSCENDENTALS

#define ucl_exp expf
#define ucl_powr powf

#else

#define ucl_exp __expf
#define ucl_powr __powf

#endif

#endif

#endif

// -------------------------------------------------------------------------
//                           NVIDIA FERMI OPENCL DEFINITIONS
// -------------------------------------------------------------------------

#ifdef FERMI_OCL

#define USE_OPENCL
#define fast_mul(X,Y) (X)*(Y)
#define ARCH 0
#define DRIVER 0
#define MEM_THREADS 32
#define THREADS_PER_ATOM 4
#define THREADS_PER_CHARGE 8
#define BLOCK_PAIR 128
#define MAX_SHARED_TYPES 11
#define BLOCK_NBOR_BUILD 128
#define BLOCK_BIO_PAIR 128

#define WARP_SIZE 32
#define PPPM_BLOCK_1D 64
#define BLOCK_CELL_2D 8
#define BLOCK_CELL_ID 128
#define MAX_BIO_SHARED_TYPES 128

#pragma OPENCL EXTENSION cl_khr_fp64: enable

#endif

// -------------------------------------------------------------------------
//                            AMD CYPRESS OPENCL DEFINITIONS
// -------------------------------------------------------------------------

#ifdef CYPRESS_OCL

#define USE_OPENCL
#define fast_mul(X,Y) (X)*(Y)
#define ARCH 0
#define DRIVER 0
#define MEM_THREADS 32
#define THREADS_PER_ATOM 4
#define THREADS_PER_CHARGE 8
#define BLOCK_PAIR 128
#define MAX_SHARED_TYPES 8
#define BLOCK_NBOR_BUILD 64
#define BLOCK_BIO_PAIR 64

#define WARP_SIZE 64
#define PPPM_BLOCK_1D 64
#define BLOCK_CELL_2D 8
#define BLOCK_CELL_ID 128
#define MAX_BIO_SHARED_TYPES 128

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#endif

// -------------------------------------------------------------------------
//                            GENERIC OPENCL DEFINITIONS
// -------------------------------------------------------------------------

#ifdef GENERIC_OCL

#define USE_OPENCL
#define fast_mul mul24
#define ARCH 0
#define DRIVER 0
#define MEM_THREADS 16
#define THREADS_PER_ATOM 1
#define THREADS_PER_CHARGE 1
#define BLOCK_PAIR 64
#define MAX_SHARED_TYPES 8
#define BLOCK_NBOR_BUILD 64
#define BLOCK_BIO_PAIR 64

#define WARP_SIZE 1
#define PPPM_BLOCK_1D 64
#define BLOCK_CELL_2D 8
#define BLOCK_CELL_ID 128
#define MAX_BIO_SHARED_TYPES 128

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#endif

// -------------------------------------------------------------------------
//                     OPENCL Stuff for All Hardware
// -------------------------------------------------------------------------
#ifdef USE_OPENCL

#define GLOBAL_ID_X get_global_id(0)
#define THREAD_ID_X get_local_id(0)
#define BLOCK_ID_X get_group_id(0)
#define BLOCK_SIZE_X get_local_size(0)
#define GLOBAL_SIZE_X get_global_size(0)
#define THREAD_ID_Y get_local_id(1)
#define BLOCK_ID_Y get_group_id(1)
#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)
#define ucl_inline inline
#define fetch_pos(i,y) x_[i]
#define fetch_q(i,y) q_[i]

#define ucl_atan atan
#define ucl_cbrt cbrt
#define ucl_ceil ceil
#define ucl_abs fabs

#ifdef _DOUBLE_DOUBLE
#define NO_HARDWARE_TRANSCENDENTALS
#endif

#ifdef NO_HARDWARE_TRANSCENDENTALS

#define ucl_exp exp
#define ucl_powr powr
#define ucl_rsqrt rsqrt
#define ucl_sqrt sqrt
#define ucl_recip(x) ((numtyp)1.0/(x))

#else

#define ucl_exp native_exp
#define ucl_powr native_powr
#define ucl_rsqrt native_rsqrt
#define ucl_sqrt native_sqrt
#define ucl_recip native_recip

#endif

#endif

// -------------------------------------------------------------------------
//                  ARCHITECTURE INDEPENDENT DEFINITIONS
// -------------------------------------------------------------------------

#define PPPM_MAX_SPLINE 8

#ifdef _DOUBLE_DOUBLE
#define numtyp double
#define numtyp2 double2
#define numtyp4 double4
#define acctyp double
#define acctyp4 double4
#endif

#ifdef _SINGLE_DOUBLE
#define numtyp float
#define numtyp2 float2
#define numtyp4 float4
#define acctyp double
#define acctyp4 double4
#endif

#ifndef numtyp
#define numtyp float
#define numtyp2 float2
#define numtyp4 float4
#define acctyp float
#define acctyp4 float4
#endif

#define EWALD_F (numtyp)1.12837917
#define EWALD_P (numtyp)0.3275911
#define A1 (numtyp)0.254829592
#define A2 (numtyp)-0.284496736
#define A3 (numtyp)1.421413741
#define A4 (numtyp)-1.453152027
#define A5 (numtyp)1.061405429

#define SBBITS 30
#define NEIGHMASK 0x3FFFFFFF
ucl_inline int sbmask(int j) { return j >> SBBITS & 3; };

