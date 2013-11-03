/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator 

   Original Version:
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov 

   See the README file in the top-level LAMMPS directory. 

   ----------------------------------------------------------------------- 

   USER-CUDA Package and associated modifications:
   https://sourceforge.net/projects/lammpscuda/ 

   Christian Trott, christian.trott@tu-ilmenau.de
   Lars Winterfeld, lars.winterfeld@tu-ilmenau.de
   Theoretical Physics II, University of Technology Ilmenau, Germany 

   See the README file in the USER-CUDA directory. 

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#include "cuda_shared.h"

extern "C" void Cuda_AtomVecGranularCuda_Init(cuda_shared_data* sdata);

extern "C" int Cuda_AtomVecGranularCuda_PackComm(cuda_shared_data* sdata,int n,int iswap,void* buf_send,int* pbc,int pbcflag,int radvary);
extern "C" int Cuda_AtomVecGranularCuda_PackCommVel(cuda_shared_data* sdata,int n,int iswap,void* buf_send,int* pbc,int pbcflag,int radvary);
extern "C" int Cuda_AtomVecGranularCuda_PackComm_Self(cuda_shared_data* sdata,int n,int iswap,int first,int* pbc,int pbcflag,int radvary);
extern "C" int Cuda_AtomVecGranularCuda_PackCommVel_Self(cuda_shared_data* sdata,int n,int iswap,int first,int* pbc,int pbcflag,int radvary);
extern "C" void Cuda_AtomVecGranularCuda_UnpackComm(cuda_shared_data* sdata,int n,int first,void* buf_recv,int iswap,int radvary);
extern "C" void Cuda_AtomVecGranularCuda_UnpackCommVel(cuda_shared_data* sdata,int n,int first,void* buf_recv,int iswap,int radvary);

/*extern "C" int Cuda_AtomVecGranularCuda_PackReverse(cuda_shared_data* sdata,int n,int first,void* buf_send);
extern "C" void Cuda_AtomVecGranularCuda_UnpackReverse(cuda_shared_data* sdata,int n,int iswap,void* buf_recv);
extern "C" void Cuda_AtomVecGranularCuda_UnpackReverse_Self(cuda_shared_data* sdata,int n,int iswap,int first);*/

extern "C" int Cuda_AtomVecGranularCuda_PackExchangeList(cuda_shared_data* sdata,int n,int dim,void* buf_send);
extern "C" int Cuda_AtomVecGranularCuda_PackExchange(cuda_shared_data* sdata,int nsend,void* buf_send,void* copylist);
extern "C" int Cuda_AtomVecGranularCuda_UnpackExchange(cuda_shared_data* sdata,int nsend,void* buf_send,void* copylist);

extern "C" int Cuda_AtomVecGranularCuda_PackBorder(cuda_shared_data* sdata,int n,int iswap,void* buf_send,int* pbc,int pbcflag);
extern "C" int Cuda_AtomVecGranularCuda_PackBorderVel(cuda_shared_data* sdata,int n,int iswap,void* buf_send,int* pbc,int pbcflag);
extern "C" int Cuda_AtomVecGranularCuda_PackBorder_Self(cuda_shared_data* sdata,int n,int iswap,int first,int* pbc,int pbcflag);
extern "C" int Cuda_AtomVecGranularCuda_PackBorderVel_Self(cuda_shared_data* sdata,int n,int iswap,int first,int* pbc,int pbcflag);
extern "C" int Cuda_AtomVecGranularCuda_UnpackBorder(cuda_shared_data* sdata,int n,int first,void* buf_recv);
extern "C" int Cuda_AtomVecGranularCuda_UnpackBorderVel(cuda_shared_data* sdata,int n,int first,void* buf_recv);
