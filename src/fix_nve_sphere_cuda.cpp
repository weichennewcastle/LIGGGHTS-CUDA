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

#include <cstdio>
#include <cstring>
#include "fix_nve_sphere_cuda.h"
#include "fix_nve_sphere_cuda_cu.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "cuda.h"
#include "cuda_modify_flags.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace FixConstCuda;

/* ---------------------------------------------------------------------- */

FixNVESphereCuda::FixNVESphereCuda(LAMMPS *lmp, int narg, char **arg) :
  FixNVECuda(lmp, narg, arg)
{
	if (strcmp(style,"nve/sphere") != 0 && narg < 3)
		error->all(FLERR,"Illegal fix nve command");
	
	time_integrate = 1;
  // error checks
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nve/sphere command");
      if (strcmp(arg[iarg+1],"dipole") == 0) error->all(FLERR,"Fix nve/sphere/cuda does not yet support dipole systems! Aborting");
      else error->all(FLERR,"Illegal fix nve/sphere command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix nve/sphere command");
  }

  if (!atom->omega_flag || !atom->torque_flag)
    error->all(FLERR,"Fix nve/sphere requires atom attributes omega, torque");

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix nve/sphere requires atom style sphere");
}

/* ---------------------------------------------------------------------- */

int FixNVESphereCuda::setmask()
{
	int mask = 0;
	mask |= INITIAL_INTEGRATE_CUDA;
	mask |= FINAL_INTEGRATE_CUDA;
	// mask |= INITIAL_INTEGRATE_RESPA_CUDA;
	// mask |= FINAL_INTEGRATE_RESPA_CUDA;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESphereCuda::init()
{
  int i,itype;

  // check that all particles are finite-size and spherical
  // no point particles allowed

  if (atom->radius_flag) {
    double *radius = atom->radius;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	if (radius[i] == 0.0)
	  error->one(FLERR,"Fix nve/sphere requires extended particles");
      }

  }

  
    dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	
	if (strcmp(update->integrate_style,"respa") == 0)
		step_respa = ((Respa *) update->integrate)->step;
		
	triggerneighsq= cuda->shared_data.atom.triggerneighsq;
    cuda->neighbor_decide_by_integrator=1;
    Cuda_FixNVESphereCuda_Init(&cuda->shared_data,dtv,dtf);
    
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVESphereCuda::initial_integrate(int vflag)
{
	if(triggerneighsq!=cuda->shared_data.atom.triggerneighsq) 
	{
		triggerneighsq= cuda->shared_data.atom.triggerneighsq;
		Cuda_FixNVESphereCuda_Init(&cuda->shared_data,dtv,dtf);
	}
	int nlocal = atom->nlocal;
	if(igroup == atom->firstgroup) nlocal = atom->nfirst;

    Cuda_FixNVESphereCuda_InitialIntegrate(& cuda->shared_data, groupbit,nlocal);	
}

/* ---------------------------------------------------------------------- */

void FixNVESphereCuda::final_integrate()
{
	int nlocal = atom->nlocal;
	if(igroup == atom->firstgroup) nlocal = atom->nfirst;
	
	Cuda_FixNVESphereCuda_FinalIntegrate(& cuda->shared_data, groupbit,nlocal);
}


/* ---------------------------------------------------------------------- */

void FixNVESphereCuda::reset_dt()
{
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	Cuda_FixNVESphereCuda_Init(&cuda->shared_data,dtv,dtf);
}
