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

#ifdef FIX_CLASS

FixStyle(nve/sphere/cuda,FixNVESphereCuda)

#else

#ifndef LMP_FIX_NVE_SPHERE_CUDA_H
#define LMP_FIX_NVE_SPHERE_CUDA_H

#include "fix_nve_cuda.h"
#include "cuda_precision.h"

namespace LAMMPS_NS {

class FixNVESphereCuda : public FixNVECuda
{
	public:
		FixNVESphereCuda(class LAMMPS *, int, char **);
		int setmask();
		virtual void init();
		virtual void initial_integrate(int);
		virtual void final_integrate();
		virtual void reset_dt();
};

}

#endif
#endif
