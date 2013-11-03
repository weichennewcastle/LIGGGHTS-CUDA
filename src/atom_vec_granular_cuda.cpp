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

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "atom_vec_granular_cuda.h"
#include "atom_vec_granular_cuda_cu.h"
#include "atom.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "cuda.h"
#include "comm.h"

using namespace LAMMPS_NS;

#define DELTA 10000
#define BUFFACTOR 1.5
#define BUFEXTRA 1000
#define NCUDAEXCHANGE 17 //nextra x y z vx vy vz tag type mask image


#define BUF_FLOAT double
/* ---------------------------------------------------------------------- */

AtomVecGranularCuda::AtomVecGranularCuda(LAMMPS *lmp, int narg, char **arg) :
  AtomVecSphere(lmp, narg, arg)
{
   maxsend=0;
   cudable=true;
   cuda_init_done=false;
   max_nsend=0;
   cu_copylist=NULL;
   copylist=NULL;
   copylist2=NULL;
}

void AtomVecGranularCuda::grow_copylist(int new_max_nsend)
{
  max_nsend=new_max_nsend;
  delete cu_copylist;
  delete [] copylist2;
  if(copylist) CudaWrapper_FreePinnedHostData((void*) copylist);
  copylist = (int*) CudaWrapper_AllocPinnedHostData(max_nsend*sizeof(int),false);
  copylist2 = new int[max_nsend];
  cu_copylist = new cCudaData<int, int, xx > (copylist, max_nsend);
}

void AtomVecGranularCuda::grow_send(int n,double** buf_send,int flag)
{
  int old_maxsend=*maxsend+BUFEXTRA;
  *maxsend = static_cast<int> (BUFFACTOR * n);
  if (flag)
  {
    if(cuda->pinned)
    {
      double* tmp = new double[old_maxsend];
      memcpy((void*) tmp,(void*) *buf_send,old_maxsend*sizeof(double));
      if(*buf_send) CudaWrapper_FreePinnedHostData((void*) (*buf_send));
      *buf_send = (double*) CudaWrapper_AllocPinnedHostData((*maxsend+BUFEXTRA)*sizeof(double),false);
      memcpy(*buf_send,tmp,old_maxsend*sizeof(double));
      delete [] tmp;	        	
    }
    else
    {
     *buf_send = (double *) 
      memory->srealloc(*buf_send,(*maxsend+BUFEXTRA)*sizeof(double),
		       "comm:buf_send");
    }
  }
  else {
   if(cuda->pinned)
    {
      if(*buf_send) CudaWrapper_FreePinnedHostData((void*) (*buf_send));
      *buf_send = (double*) CudaWrapper_AllocPinnedHostData((*maxsend+BUFEXTRA)*sizeof(double),false);
    }
    else
    {
      memory->sfree(*buf_send);
      *buf_send = (double *) memory->smalloc((*maxsend+BUFEXTRA)*sizeof(double),
					  "comm:buf_send");
    }
  }
}

void AtomVecGranularCuda::grow_both(int n)
{
  if(cuda->finished_setup)
  cuda->downloadAll();	
  AtomVecSphere::grow(n);
  if(cuda->finished_setup)
  {
    cuda->checkResize();
    cuda->uploadAll();
  }
}

int AtomVecGranularCuda::pack_comm(int n, int* iswap, double *buf,
			     int pbc_flag, int *pbc)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	return AtomVecSphere::pack_comm(n,iswap,buf,pbc_flag,pbc);
  //	printf("Parameters: %i %i %i\n",cuda->self_comm,*iswap,((int*) buf)[*iswap]);
  	int m;
  	if(cuda->self_comm)
  		m = Cuda_AtomVecGranularCuda_PackComm_Self(&cuda->shared_data,n,*iswap,((int*) buf)[*iswap],pbc,pbc_flag,radvary);
  	else
	    m = Cuda_AtomVecGranularCuda_PackComm(&cuda->shared_data,n,*iswap,(void*) buf,pbc,pbc_flag,radvary);
	if((sizeof(X_FLOAT)!=sizeof(double)) && m)
	  m=(m+1)*sizeof(X_FLOAT)/sizeof(double);
	return m;
}

int AtomVecGranularCuda::pack_comm_vel(int n, int* iswap, double *buf,
			     int pbc_flag, int *pbc)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	return AtomVecSphere::pack_comm_vel(n,iswap,buf,pbc_flag,pbc);
   //	printf("ParametersVel: %i %i %i\n",cuda->self_comm,*iswap,((int*) buf)[*iswap]);
  	
  	int m;
  	
	if(cuda->self_comm)
	m = Cuda_AtomVecGranularCuda_PackCommVel_Self(&cuda->shared_data,n,*iswap,((int*) buf)[*iswap],pbc,pbc_flag,radvary);
	else
	m = Cuda_AtomVecGranularCuda_PackCommVel(&cuda->shared_data,n,*iswap,(void*) buf,pbc,pbc_flag,radvary);
	if((sizeof(X_FLOAT)!=sizeof(double)) && m)
	  m=(m+1)*sizeof(X_FLOAT)/sizeof(double);
	return m;
}
/* ---------------------------------------------------------------------- */

void AtomVecGranularCuda::unpack_comm(int n, int first, double *buf)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	 {AtomVecSphere::unpack_comm(n,first,buf); return;}
  	
  if(not cuda->self_comm)
  Cuda_AtomVecGranularCuda_UnpackComm(&cuda->shared_data,n,first,(void*)buf,-1,radvary);
}

void AtomVecGranularCuda::unpack_comm_vel(int n, int first, double *buf)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	 {AtomVecSphere::unpack_comm_vel(n,first,buf); return;}

  if(not cuda->self_comm)
  Cuda_AtomVecGranularCuda_UnpackCommVel(&cuda->shared_data,n,first,(void*)buf,-1,radvary);
}
/* ---------------------------------------------------------------------- */

int AtomVecGranularCuda::pack_reverse(int n, int first, double *buf)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	return AtomVecSphere::pack_reverse(n,first,buf);

  int i,m,last;
  cuda->cu_f->download();

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
  }
  cuda->cu_f->download();
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecGranularCuda::unpack_reverse(int n, int *list, double *buf)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	{AtomVecSphere::unpack_reverse(n,list,buf); return;}

  int i,j,m;
  cuda->cu_f->download();

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }
  cuda->cu_f->upload();
}

/* ---------------------------------------------------------------------- */

int AtomVecGranularCuda::pack_border(int n, int *iswap, double *buf,
			       int pbc_flag, int *pbc)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	return AtomVecSphere::pack_border(n,iswap,buf,pbc_flag,pbc);
  
  int m = Cuda_AtomVecGranularCuda_PackBorder(&cuda->shared_data,n,*iswap,(void*) buf,pbc,pbc_flag);

  return m;
}

int AtomVecGranularCuda::pack_border_vel(int n, int *iswap, double *buf,
			       int pbc_flag, int *pbc)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	return AtomVecSphere::pack_border_vel(n,iswap,buf,pbc_flag,pbc);
  
  int m = Cuda_AtomVecGranularCuda_PackBorderVel(&cuda->shared_data,n,*iswap,(void*) buf,pbc,pbc_flag);

  return m;
}
/* ---------------------------------------------------------------------- */

void AtomVecGranularCuda::unpack_border(int n, int first, double *buf)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	 {AtomVecSphere::unpack_border(n,first,buf); return;}
  while(atom->nghost+atom->nlocal+n>=cuda->shared_data.atom.nmax) 
  {
  	grow_both(0);
  }
  int flag=Cuda_AtomVecGranularCuda_UnpackBorder(&cuda->shared_data,n,first,(void*)buf);
  if(flag) {printf(" # CUDA: Error: Failed to unpack Border atoms (This might be a bug).\n");}

}

void AtomVecGranularCuda::unpack_border_vel(int n, int first, double *buf)
{
  if(not cuda->finished_setup || cuda->oncpu)
  	 {AtomVecSphere::unpack_border_vel(n,first,buf); return;}
  while(atom->nghost+atom->nlocal+n>=cuda->shared_data.atom.nmax) 
  {
  	grow_both(0);
  }
  int flag=Cuda_AtomVecGranularCuda_UnpackBorderVel(&cuda->shared_data,n,first,(void*)buf);
  if(flag) {printf(" # CUDA: Error: Failed to unpack Border atoms (This might be a bug).\n");}
}
/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them 
------------------------------------------------------------------------- */


int AtomVecGranularCuda::pack_exchange(int dim, double *buf)
{
  if(cuda->oncpu)
  	return AtomVecSphere::pack_exchange(dim,buf);

  if(not cuda_init_done)
  {
  	Cuda_AtomVecGranularCuda_Init(&cuda->shared_data);
  	cuda_init_done=true;
  }

  double** buf_pointer=(double**) buf;
  if(*maxsend<atom->nghost || *buf_pointer==NULL)
  {
  	grow_send(atom->nghost>*maxsend?atom->nghost:*maxsend,buf_pointer,0);
  	*maxsend=atom->nghost>*maxsend?atom->nghost:*maxsend;
  }

  if(max_nsend==0) grow_copylist(200);
  int nsend_atoms = Cuda_AtomVecGranularCuda_PackExchangeList(&cuda->shared_data,*maxsend,dim,*buf_pointer);

  if(nsend_atoms>max_nsend) grow_copylist(nsend_atoms+100);
  if(nsend_atoms*NCUDAEXCHANGE>*maxsend) 
  {
  	grow_send((int) (nsend_atoms+100)*NCUDAEXCHANGE,buf_pointer,0);
  	Cuda_AtomVecGranularCuda_PackExchangeList(&cuda->shared_data,*maxsend,dim,*buf_pointer);
  }

  int nlocal=atom->nlocal-nsend_atoms;
  
  for(int i=0;i<nsend_atoms;i++) copylist2[i]=1;
  for(int j=1;j<nsend_atoms+1;j++)
  {
  	int i = static_cast <int> ((*buf_pointer)[j]);
  	if(i>=nlocal) copylist2[i-nlocal]=-1;
  }
  
  int actpos=0;
  for(int j=1;j<nsend_atoms+1;j++)
  {
  	int i = static_cast <int> ((*buf_pointer)[j]);
  	if(i<nlocal) 
  	{
  	  while(copylist2[actpos]==-1) actpos++;
    	  copylist[j-1]=nlocal+actpos;
  	  actpos++;
  	}
  }
  cu_copylist->upload();
  
  cuda->shared_data.atom.nlocal=nlocal;
 
  int m = Cuda_AtomVecGranularCuda_PackExchange(&cuda->shared_data,nsend_atoms,*buf_pointer,cu_copylist->dev_data());
  timespec time1,time2;
  clock_gettime(CLOCK_REALTIME,&time1);
 
  double* buf_p=*buf_pointer;
  for(int j=0;j<nsend_atoms;j++)
  {
    int i=static_cast <int> (buf_p[j+1]);
    int nextra=0;
    int k;
 
    if (atom->nextra_grow)
      for (int iextra = 0; iextra < atom->nextra_grow; iextra++) 
      {
        int dm= modify->fix[atom->extra_grow[iextra]]->pack_exchange(i,&buf_p[m]);
        m+=dm;
  		nextra+=dm;
        if(i<nlocal)modify->fix[atom->extra_grow[iextra]]->copy_arrays(copylist[j],i);
    	if(m>*maxsend) {grow_send(m,buf_pointer,1); buf_p=*buf_pointer;}
      }

    if(i<nlocal)AtomVecSphere::copy(copylist[j],i,1);  
    (*buf_pointer)[j+1] = nextra;
  }
	  
	  clock_gettime(CLOCK_REALTIME,&time2);
	  cuda->shared_data.cuda_timings.comm_exchange_cpu_pack+=
        time2.tv_sec-time1.tv_sec+1.0*(time2.tv_nsec-time1.tv_nsec)/1000000000;

  (*buf_pointer)[0] = nsend_atoms;
  atom->nlocal-=nsend_atoms;
  cuda->shared_data.atom.update_nlocal=2;
 //printf("End Pack Exchange\n");
  if(m==1) return 0;
  return m; 
}

/* ---------------------------------------------------------------------- */

int AtomVecGranularCuda::unpack_exchange(double *buf)
{
  if(cuda->oncpu)
  	return AtomVecSphere::unpack_exchange(buf);

  double *sublo,*subhi;

  int dim=cuda->shared_data.exchange_dim;
  if(domain->box_change) 
  Cuda_AtomVecGranularCuda_Init(&cuda->shared_data);
  if (domain->triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  int mfirst=0;
  for(int pi=0;pi<(comm->procgrid[dim]>2?2:1);pi++)
  {
  int nlocal = atom->nlocal;
  int nsend_atoms=static_cast<int> (buf[0]);
  if(nsend_atoms>max_nsend) grow_copylist(nsend_atoms+100);
 
  if (nlocal+nsend_atoms+atom->nghost>=atom->nmax) grow_both(nlocal+nsend_atoms*2+atom->nghost); //ensure there is enough space on device to unpack data
  int naccept = Cuda_AtomVecGranularCuda_UnpackExchange(&cuda->shared_data,nsend_atoms,buf,cu_copylist->dev_data());
  cu_copylist->download();
  int m = nsend_atoms*NCUDAEXCHANGE + 1;
  nlocal+=naccept;

  timespec time1,time2;
  clock_gettime(CLOCK_REALTIME,&time1);

  for(int j=0;j<nsend_atoms;j++)
  {
    if(copylist[j]>-1)
    {
 	  int k;
	  int i=copylist[j];
   	
  	  if (atom->nextra_grow)
        for (int iextra = 0; iextra < atom->nextra_grow; iextra++) 
      				m += modify->fix[atom->extra_grow[iextra]]->
					unpack_exchange(i,&buf[m]);
    	
    }
    else 
    m+=static_cast <int> (buf[j+1]);
  }
	  
	  clock_gettime(CLOCK_REALTIME,&time2);
	  cuda->shared_data.cuda_timings.comm_exchange_cpu_pack+=
        time2.tv_sec-time1.tv_sec+1.0*(time2.tv_nsec-time1.tv_nsec)/1000000000;

  cuda->shared_data.atom.nlocal=nlocal;
  cuda->shared_data.atom.update_nlocal=2;
  atom->nlocal=nlocal;
  mfirst+=m;
  buf=&buf[m];
  }
  return mfirst;
}



