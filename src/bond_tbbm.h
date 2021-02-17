/* ----------------------------------------------------------------------
   LIGGGHTS - LAMMPS Improved for General Granular and Granular Heat
   Transfer Simulations

   LIGGGHTS is part of the CFDEMproject
   www.liggghts.com | www.cfdem.com

   Christoph Kloss, christoph.kloss@cfdem.com
   Copyright 2009-2012 JKU Linz
   Copyright 2012-     DCS Computing GmbH, Linz

   LIGGGHTS is based on LAMMPS
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(tbbm,BondTBBM)

#else

#ifndef LMP_BOND_TBBM_H
#define LMP_BOND_TBBM_H

#include "stdio.h"
#include "bond.h"

namespace LAMMPS_NS {

class BondTBBM : public Bond {
 public:
  BondTBBM(class LAMMPS *);
  ~BondTBBM();
  void init_style();
  void compute(int, int);
  void coeff(int, char **);
  void strength();
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  //double single(int, double, int, int);
  double single(int, double, int, int, double &);

  double getMinDt();

 protected:
  double *Y,*G;
  double *compression_break,*shear_break,*tensile_break;
  double *CoV_compression,*CoV_tensile,*CoV_shear;
  bool strength_flag;
  double *ro;
  double *random_number;  
  double *N;
  double *Strength_c;
  double *Strength_t;
  double *Strength_s;
  
  void allocate();
 private:
  int nStrengthBondalloc;
  int nFile;

;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

*/
