// **************************************************************************
//                                   lj.cu
//                             -------------------
//                           W. Michael Brown (ORNL)
//
//  Device code for acceleration of the lj/cut pair style
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                : 
//    email                : brownw@ornl.gov
// ***************************************************************************/

#ifdef NV_KERNEL
#include "lal_aux_fun1.h"
texture<float4> pos_tex;
#ifndef _DOUBLE_DOUBLE
ucl_inline float4 fetch_pos(const int& i, const float4 *pos) 
  { return tex1Dfetch(pos_tex, i); }
#endif
#endif

__kernel void kernel_pair(__global numtyp4 *x_, __global numtyp4 *lj1,
                          __global numtyp4* lj3, const int lj_types, 
                          __global numtyp *sp_lj_in, __global int *dev_nbor, 
                          __global int *dev_packed, __global acctyp4 *ans,
                          __global acctyp *engv, const int eflag, 
                          const int vflag, const int inum,
                          const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp sp_lj[4];
  sp_lj[0]=sp_lj_in[0];
  sp_lj[1]=sp_lj_in[1];
  sp_lj[2]=sp_lj_in[2];
  sp_lj[3]=sp_lj_in[3];

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;
  
  if (ii<inum) {
    __global int *nbor, *list_end;
    int i, numj, n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,list_end,nbor);
  
    numtyp4 ix=fetch_pos(i,x_); //x_[i];
    int itype=ix.w;

    numtyp factor_lj;
    for ( ; nbor<list_end; nbor+=n_stride) {
  
      int j=*nbor;
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx=fetch_pos(j,x_); //x_[j];
      int jtype=jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp r2inv = delx*delx+dely*dely+delz*delz;
        
      int mtype=itype*lj_types+jtype;
      if (r2inv<lj1[mtype].z) {
        r2inv=ucl_recip(r2inv);
        numtyp r6inv = r2inv*r2inv*r2inv;
        numtyp force = r2inv*r6inv*(lj1[mtype].x*r6inv-lj1[mtype].y);
        force*=factor_lj;
      
        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (eflag>0) {
          numtyp e=r6inv*(lj3[mtype].x*r6inv-lj3[mtype].y);
          energy+=factor_lj*(e-lj3[mtype].z); 
        }
        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

__kernel void kernel_pair_fast(__global numtyp4 *x_, __global numtyp4 *lj1_in,
                               __global numtyp4* lj3_in, 
                               __global numtyp* sp_lj_in, 
                               __global int *dev_nbor, __global int *dev_packed, 
                               __global acctyp4 *ans, __global acctyp *engv, 
                               const int eflag, const int vflag, const int inum, 
                               const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);
  
  __local numtyp4 lj1[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp4 lj3[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp sp_lj[4];
  if (tid<4)
    sp_lj[tid]=sp_lj_in[tid];
  if (tid<MAX_SHARED_TYPES*MAX_SHARED_TYPES) {
    lj1[tid]=lj1_in[tid];
    if (eflag>0)
      lj3[tid]=lj3_in[tid];
  }
  
  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  __syncthreads();
  
  if (ii<inum) {
    __global int *nbor, *list_end;
    int i, numj, n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,list_end,nbor);

    numtyp4 ix=fetch_pos(i,x_); //x_[i];
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);

    numtyp factor_lj;
    for ( ; nbor<list_end; nbor+=n_stride) {
  
      int j=*nbor;
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx=fetch_pos(j,x_); //x_[j];
      int mtype=itype+jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp r2inv = delx*delx+dely*dely+delz*delz;
        
      if (r2inv<lj1[mtype].z) {
        r2inv=ucl_recip(r2inv);
        numtyp r6inv = r2inv*r2inv*r2inv;
        numtyp force = factor_lj*r2inv*r6inv*(lj1[mtype].x*r6inv-lj1[mtype].y);
      
        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (eflag>0) {
          numtyp e=r6inv*(lj3[mtype].x*r6inv-lj3[mtype].y);
          energy+=factor_lj*(e-lj3[mtype].z); 
        }
        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

