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

__global__ void Cuda_AtomVecGranularCuda_PackComm_Kernel(int* sendlist,int n,int maxlistlength,int iswap,X_FLOAT dx,X_FLOAT dy,X_FLOAT dz)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
    int j=list[i];
  
    ((X_FLOAT*) _buffer)[i]=_x[j] + dx;
    ((X_FLOAT*) _buffer)[i+1*n] = _x[j+_nmax] + dy;
    ((X_FLOAT*) _buffer)[i+2*n] = _x[j+2*_nmax] + dz;
  }
}

__global__ void Cuda_AtomVecGranularCuda_PackCommVel_Kernel(int* sendlist,int n,int maxlistlength,int iswap,X_FLOAT dx,X_FLOAT dy,X_FLOAT dz)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
    int j=list[i];
  
    ((X_FLOAT*) _buffer)[i]=_x[j] + dx;
    ((X_FLOAT*) _buffer)[i+1*n] = _x[j+_nmax] + dy;
    ((X_FLOAT*) _buffer)[i+2*n] = _x[j+2*_nmax] + dz;
    ((X_FLOAT*) _buffer)[i+3*n]=_v[j];
    ((X_FLOAT*) _buffer)[i+4*n] = _v[j+_nmax];
    ((X_FLOAT*) _buffer)[i+5*n] = _v[j+2*_nmax];
  }
}

__global__ void Cuda_AtomVecGranularCuda_PackComm_Self_Kernel(int* sendlist,int n,int maxlistlength,int iswap,X_FLOAT dx,X_FLOAT dy,X_FLOAT dz,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
    int j=list[i];
  
    _x[i+first]=_x[j] + dx;
    _x[i+first+_nmax] = _x[j+_nmax] + dy;
    _x[i+first+2*_nmax] = _x[j+2*_nmax] + dz;
  }
  
}

__global__ void Cuda_AtomVecGranularCuda_PackCommVel_Self_Kernel(int* sendlist,int n,int maxlistlength,int iswap,X_FLOAT dx,X_FLOAT dy,X_FLOAT dz,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
    int j=list[i];
  
    _x[i+first]=_x[j] + dx;
    _x[i+first+_nmax] = _x[j+_nmax] + dy;
    _x[i+first+2*_nmax] = _x[j+2*_nmax] + dz;
    _v[i+first]=_v[j];
    _v[i+first+_nmax] = _v[j+_nmax];
    _v[i+first+2*_nmax] = _v[j+2*_nmax];
  }
  
}

__global__ void Cuda_AtomVecGranularCuda_UnpackComm_Kernel(int n,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  if(i<n)
  {
  _x[i+first]=((X_FLOAT*) _buffer)[i];
  _x[i+first+_nmax]=((X_FLOAT*) _buffer)[i+1*n];
  _x[i+first+2*_nmax]=((X_FLOAT*) _buffer)[i+2*n];
  }
}

__global__ void Cuda_AtomVecGranularCuda_UnpackCommVel_Kernel(int n,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  if(i<n)
  {
  _x[i+first]=((X_FLOAT*) _buffer)[i];
  _x[i+first+_nmax]=((X_FLOAT*) _buffer)[i+1*n];
  _x[i+first+2*_nmax]=((X_FLOAT*) _buffer)[i+2*n];
  _v[i+first]=((X_FLOAT*) _buffer)[i+3*n];
  _v[i+first+_nmax]=((X_FLOAT*) _buffer)[i+4*n];
  _v[i+first+2*_nmax]=((X_FLOAT*) _buffer)[i+5*n];
  }
}

__global__ void Cuda_AtomVecGranularCuda_PackReverse_Kernel(int n,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  if(i<n)
  {
  ((F_FLOAT*) _buffer)[i]=_f[i+first];
  ((F_FLOAT*) _buffer)[i+n] = _f[i+first+_nmax];
  ((F_FLOAT*) _buffer)[i+2*n] = _f[i+first+2*_nmax];
  }
  
}

__global__ void Cuda_AtomVecGranularCuda_UnpackReverse_Kernel(int* sendlist,int n,int maxlistlength,int iswap)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
  int j=list[i];
  _f[j]+=((F_FLOAT*)_buffer)[i];
  _f[j+_nmax]+=((F_FLOAT*) _buffer)[i+n];
  _f[j+2*_nmax]+=((F_FLOAT*) _buffer)[i+2*n];
  }
  
}

__global__ void Cuda_AtomVecGranularCuda_UnpackReverse_Self_Kernel(int* sendlist,int n,int maxlistlength,int iswap,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
  int j=list[i];
  
  _f[j]+=_f[i+first];
  _f[j+_nmax]+=_f[i+first+_nmax];
  _f[j+2*_nmax]+=_f[i+first+2*_nmax];
  }
  
}


__global__ void Cuda_AtomVecGranularCuda_PackExchange_Kernel(int n,int dim)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  double* buf=(double*) _buffer;
  buf=&buf[1];
  
  if(i<_nlocal)
  {
  	
    if (static_cast <double> (_x[i+dim*_nmax]) < _sublo[dim] || static_cast <double> (_x[i+dim*_nmax]) > _subhi[dim]) //only send if atom is really outside (in original lammps its compared vs >=_subhi
    {
	  int j=atomicAdd((int*)_buffer,1);
	  if(NCUDAEXCHANGE*(j+1)<n)
	  {
	    buf=&buf[NCUDAEXCHANGE*j];
	    buf[0]=i;
	    buf[1]=static_cast <double> (_x[i]);
	    buf[2]=static_cast <double> (_x[i+_nmax]);
	    buf[3]=static_cast <double> (_x[i+2*_nmax]);
	    buf[4]=_v[i];
	    buf[5]=_v[i+_nmax];
	    buf[6]=_v[i+2*_nmax];
	    int atag=_tag[i];
	    buf[7]=atag<0?-atag:atag;
	    _tag[i]=atag<0?atag:-atag;
	    buf[8]=_type[i];
	    buf[9]=_mask[i];
	    buf[10]=_image[i];
	  }
    }
  }
}

__global__ void Cuda_AtomVecGranularCuda_PackExchange_FillExchanges_Kernel()
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;

  double* buf=(double*) _buffer;
  int nsend=((int*) _buffer)[0];
  buf=&buf[1];

  int j=0;
  int k=_nlocal;
  int sendlist=-1;
  int replacelist=-1;
  if(i<nsend)
  {
    j=static_cast <int> (buf[NCUDAEXCHANGE*i]);
    if(j<_nlocal-nsend)
    for(int l=0;l<=i;l++)
    if(static_cast <int> (buf[NCUDAEXCHANGE*l])<_nlocal-nsend) sendlist++;
  }
  __syncthreads();
  if((i<nsend)&&(j<_nlocal-nsend))
  {  
    for(int l=_nlocal-1;l>=_nlocal-nsend;l--)
    {
      if(_tag[l]>0) {replacelist++; if(replacelist==sendlist) k=l;}
    }
  }
  
  __syncthreads();
  if((j<_nlocal-nsend)&&(i<nsend))
  {
    _x[j]=_x[k];
    _x[j+_nmax]=_x[k+_nmax];
    _x[j+2*_nmax]=_x[k+2*_nmax];  
    _v[j]=_v[k];
    _v[j+_nmax]=_v[k+_nmax];
    _v[j+2*_nmax]=_v[k+2*_nmax];  
    _tag[j]=_tag[k];
    _type[j]=_type[k];
    _mask[j]=_mask[k];
    _image[j]=_image[k];
  } 
}

__global__ void Cuda_AtomVecGranularCuda_UnpackExchange_Kernel(int dim)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  double* buf=(double*) _buffer;
  buf=&buf[1];
  int n_add=static_cast<int> (((double*)_buffer)[0]);
  
  if(i<n_add)
  {
	    buf=&buf[NCUDAEXCHANGE*i];
 	    if(buf[1+dim]>=_sublo[dim] && buf[1+dim]<_subhi[dim])
	   	{
	   	  int j=atomicAdd(_flag,1)+_nlocal;
	   	
	      _x[j]=buf[1];
	      _x[j+_nmax]=buf[2];
	      _x[j+2*_nmax]=buf[3];
	      _v[j]=buf[4];
	      _v[j+_nmax]=buf[5];
	      _v[j+2*_nmax]=buf[6];
	      _tag[j]=buf[7];
	      _type[j]=buf[8];
	      _mask[j]=buf[9];
	      _image[j]=buf[10];  
	   	}
  }
}

__global__ void Cuda_AtomVecGranularCuda_PackBorder_Kernel(int* sendlist,int n,int maxlistlength,int iswap,X_FLOAT dx,X_FLOAT dy,X_FLOAT dz)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
    int j=list[i];
  
    ((X_FLOAT*) _buffer)[i]=_x[j] + dx;
    ((X_FLOAT*) _buffer)[i+1*n] = _x[j+_nmax] + dy;
    ((X_FLOAT*) _buffer)[i+2*n] = _x[j+2*_nmax] + dz;
    ((X_FLOAT*) _buffer)[i+3*n] = _tag[j];
    ((X_FLOAT*) _buffer)[i+4*n] = _type[j];
    ((X_FLOAT*) _buffer)[i+5*n] = _mask[j];
  }
  
}

__global__ void Cuda_AtomVecGranularCuda_PackBorderVel_Kernel(int* sendlist,int n,int maxlistlength,int iswap,X_FLOAT dx,X_FLOAT dy,X_FLOAT dz)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
    int j=list[i];
  
    ((X_FLOAT*) _buffer)[i]=_x[j] + dx;
    ((X_FLOAT*) _buffer)[i+1*n] = _x[j+_nmax] + dy;
    ((X_FLOAT*) _buffer)[i+2*n] = _x[j+2*_nmax] + dz;
    ((X_FLOAT*) _buffer)[i+3*n] = _tag[j];
    ((X_FLOAT*) _buffer)[i+4*n] = _type[j];
    ((X_FLOAT*) _buffer)[i+5*n] = _mask[j];
    ((X_FLOAT*) _buffer)[i+6*n] = _v[j];
    ((X_FLOAT*) _buffer)[i+7*n] = _v[j+_nmax];
    ((X_FLOAT*) _buffer)[i+8*n] = _v[j+2*_nmax];
  }
  
}

__global__ void Cuda_AtomVecGranularCuda_PackBorder_Self_Kernel(int* sendlist,int n,int maxlistlength,int iswap,X_FLOAT dx,X_FLOAT dy,X_FLOAT dz,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  int* list=sendlist+iswap*maxlistlength;
  if(i<n)
  {
    int j=list[i];
  
    _x[i+first]=_x[j] + dx;
    _x[i+first+_nmax] = _x[j+_nmax] + dy;
    _x[i+first+2*_nmax] = _x[j+2*_nmax] + dz;
	_tag[i+first] = _tag[j];
	_type[i+first] = _type[j];
	_mask[i+first] = _mask[j];
  }
}

__global__ void Cuda_AtomVecGranularCuda_UnpackBorder_Kernel(int n,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  if(i<n)
  {
  	if(i+first<_nmax)
  	{
  	  _x[i+first]=((X_FLOAT*) _buffer)[i];
  	  _x[i+first+_nmax]=((X_FLOAT*) _buffer)[i+1*n];
  	  _x[i+first+2*_nmax]=((X_FLOAT*) _buffer)[i+2*n];
  	  _tag[i+first] = static_cast<int> (((X_FLOAT*) _buffer)[i+3*n]);
  	  _type[i+first] = static_cast<int> (((X_FLOAT*) _buffer)[i+4*n]);
  	  _mask[i+first] = static_cast<int> (((X_FLOAT*) _buffer)[i+5*n]);
  	}
  	else
  	{
  	  _flag[0]=1;
  	}
  }
}

__global__ void Cuda_AtomVecGranularCuda_UnpackBorderVel_Kernel(int n,int first)
{
  int i=(blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x+threadIdx.x;
  if(i<n)
  {
  	if(i+first<_nmax)
  	{
  	  _x[i+first]=((X_FLOAT*) _buffer)[i];
  	  _x[i+first+_nmax]=((X_FLOAT*) _buffer)[i+1*n];
  	  _x[i+first+2*_nmax]=((X_FLOAT*) _buffer)[i+2*n];
  	  _tag[i+first] = static_cast<int> (((X_FLOAT*) _buffer)[i+3*n]);
  	  _type[i+first] = static_cast<int> (((X_FLOAT*) _buffer)[i+4*n]);
  	  _mask[i+first] = static_cast<int> (((X_FLOAT*) _buffer)[i+5*n]);
  	  _v[i+first]=((X_FLOAT*) _buffer)[i+6*n];
  	  _v[i+first+_nmax]=((X_FLOAT*) _buffer)[i+7*n];
  	  _v[i+first+2*_nmax]=((X_FLOAT*) _buffer)[i+8*n];
  	}
  	else
  	{
  	  _flag[0]=1;
  	}
  }
}
