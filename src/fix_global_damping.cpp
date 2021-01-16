/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "fix_global_damping.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

FixGlobalDamping::FixGlobalDamping(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal fix globaldamping command");
  damp = force->numeric(FLERR,arg[3]);
  fprintf(screen,"global damp value = %f\n ",damp);
  
}

/* ---------------------------------------------------------------------- */

FixGlobalDamping::~FixGlobalDamping()
{
}

/* ---------------------------------------------------------------------- */

int FixGlobalDamping::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGlobalDamping::init()
{
}

/* ---------------------------------------------------------------------- */

void FixGlobalDamping::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
  
}

/* ---------------------------------------------------------------------- */

void FixGlobalDamping::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixGlobalDamping::post_force(int vflag)
{
  double **f = atom->f;
  double **v = atom->v;
  double **omega = atom->omega;
  double **torque = atom->torque;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;


  for (int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {

      if (v[i][0] != 0) f[i][0] += -damp*fabs(f[i][0])*v[i][0]/fabs(v[i][0]);            
      if (v[i][1] != 0) f[i][1] += -damp*fabs(f[i][1])*v[i][1]/fabs(v[i][1]);
      if (v[i][2] != 0) f[i][2] += -damp*fabs(f[i][2])*v[i][2]/fabs(v[i][2]);
      if (omega[i][0] != 0) torque[i][0] += -damp*fabs(torque[i][0])*omega[i][0]/fabs(omega[i][0]);
      if (omega[i][1] != 0) torque[i][1] += -damp*fabs(torque[i][1])*omega[i][1]/fabs(omega[i][1]);
      if (omega[i][2] != 0) torque[i][2] += -damp*fabs(torque[i][2])*omega[i][2]/fabs(omega[i][2]);        
    }
  }
  
}

/* ---------------------------------------------------------------------- */

void FixGlobalDamping::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixGlobalDamping::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixGlobalDamping::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}
