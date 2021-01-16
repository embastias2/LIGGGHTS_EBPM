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

#include <cmath>
#include <stdlib.h>
#include "bond_tbbm.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "fix_property_atom.h"
#include "error.h"
#include "update.h"
#include "vector_liggghts.h"
#include <math.h>
#include <ctime>


using namespace LAMMPS_NS;

/*NP
large TODO list for granular bonds:  (could be a diploma thesis?)

+ need a better dissipative formulation than the hardcoded
  'dissipate' value which produces plastic deformation
  need some vel-dependant damping
MS This has been done with a velocity based version and a force based version

+ need to carefully debug and validate this bond style
  valiation against fix rigid
MS Validation has been done against the cantilever beam and beam theory.

+ check whether run this bond style w/ or w/o gran pair style active,
  (neigh_modify command)
Needs grain to handle torque

+ need to store bond radii per particle, not per type
+ parallel debugging and testing not yet done
+ need evtally implemetation
*/

/* Matt Schramm edits for bond dampening --> MS */
/* Yu Guo edits for bond dampening --> YG */
/* D. Kramolis edits for domain end detection and other bug fixes --> KRAMOLIS */
/* Emmanuel Bastias edits for breakstyle comp-tens --> EB */

/* ---------------------------------------------------------------------- */

BondTBBM::BondTBBM(LAMMPS *lmp) : Bond(lmp)
{
    // we need 13 history values - the 6 for the forces, 6 for torques from the last time-step and 1 for initial bond length
    n_granhistory(23);
    /*	NP
       number of entries in bondhistlist. bondhistlist[number of bond][number of value (from 0 to number given here)]
       so with this number you can modify how many pieces of information you savae with every bond
       following dependencies and processes for saving,copying,growing the bondhistlist:
    */
     
    /* NP
       gibt groesse der gespeicherten werte  pro bond wieder 
       neighbor.cpp:       memory->create(bondhistlist,maxbond,atom->n_bondhist,"neigh:bondhistlist");
       neigh_bond.cpp:     memory->grow(bondhistlist,maxbond,atom->n_bondhist,"neighbor:bondhistlist");
       bond.cpp: void Bond::n_granhistory(int nhist) {ngranhistory = nhist;     atom->n_bondhist = ngranhistory; if(){FLERR}}
       atom_vec_bond_gran.cpp:  memory->grow(atom->bond_hist,nmax,atom->bond_per_atom,atom->n_bondhist,"atom:bond_hist");
     */
    if(!atom->style_match("bond/gran"))
      error->all(FLERR,"A granular bond style can only be used together with atom style bond/gran");
    if(comm->me == 0)
        error->warning(FLERR,"Bond granular: This is a beta version - be careful!");
    
}

/* ---------------------------------------------------------------------- */

BondTBBM::~BondTBBM()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(ro);
    memory->destroy(Y);
    memory->destroy(G);
    memory->destroy(compression_break);
    memory->destroy(tensile_break);
    memory->destroy(shear_break);
    memory->destroy(CoV_compression);
    memory->destroy(CoV_tensile);
    memory->destroy(CoV_shear);

    delete [] random_number;
    delete [] N;
    delete [] Strength_c;
    delete [] Strength_t;
    delete [] Strength_s;

  }
}

/* ---------------------------------------------------------------------- */

void  BondTBBM::init_style()
{
  
}

/* ---------------------------------------------------------------------- */
#define SIGNUM_DOUBLE(x) (((x) > 0.0) ? 1.0 : (((x) < 0.0) ? -1.0 : 0.0))
void BondTBBM::compute(int eflag, int vflag)
{
  
  double T[3][3];
  double Z[3] = {0,0,1};
  double Tinv[3][3];
  double det_T,det_Tinv;
  double ug[12];
  double u[12];
  double K[12][12] = {0.0};
  double dF_local[12];
  double F[12];
  double comp_a,comp_b,comp_stress;
  double tens_a,tens_b,tens_stress;
  double shear_stress;
  bool comp_break,tens_break,tau_break;
  double rsq,r,rinv,rsqinv,r2,r3,r2inv,r3inv,phi,phi_pinv;

  int i1,i2,n,type;
  double delx,dely,delz; // ,ebond;
  double A,Ainv,Ginv;
  
  double I,J;
  double rout;
  double bondLength,bondLengthinv,bondLengthinv2,bondLengthinv3;
  double bondLength2,bondLength3;
  double compression_strength;
  double tensile_strength;
  double shear_strength;
  

  // ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double *radius = atom->radius;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  double **omega = atom->omega;
  double **torque = atom->torque;

  // int *tag = atom->tag; // tag of atom is their ID number
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  double dt = update->dt;
  
 
  for (n = 0; n < nbondlist; n++) { // Loop through Bond list
    //1st check if bond is broken,
    if(bondlist[n][3])
    {
      //printf("bond %d allready broken\n",n);
      bondhistlist[n][17] = 1.0;
      bondhistlist[n][18] = 1.0;
      bondhistlist[n][19] = 1.0;
      continue;
    }

    i1 = bondlist[n][0]; // Sphere 1
    i2 = bondlist[n][1]; // Sphere 2

    //2nd check if bond overlap the box-borders
    // KRAMOLIS do not use neighbor cutoff - it is usually set as 4 * atom radius or more (bugfix - correct sign - bonds were breaking inside domain far away from wall, added info about broken bonds) 
   // should be rather bond_skin or bond_length or radius, enable periodic detection
   //2nd check if bond overlap the box-borders
    double maxoverlap = radius[i1];  // max overlap is diameter - but position is in the middle of atom
    if ((x[i1][0]<(domain->boxlo[0]-maxoverlap)) && (domain->xperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at xmin\n");
#     endif
      continue;
    } else if ((x[i1][0]>(domain->boxhi[0]+maxoverlap)) && (domain->xperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at xmax\n");
#     endif
      continue;
    } else if ((x[i1][1]<(domain->boxlo[1]-maxoverlap)) && (domain->yperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at ymin\n");
#     endif
      continue;
    } else if ((x[i1][1]>(domain->boxhi[1]+maxoverlap)) && (domain->yperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at ymax\n");
#     endif
      continue;
    } else if ((x[i1][2]<(domain->boxlo[2]-maxoverlap)) && (domain->zperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at zmin\n");
#     endif
      continue;
    } else if ((x[i1][2]>(domain->boxhi[2]+maxoverlap)) && (domain->zperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at zmax\n");
#     endif
      continue;
    }
    maxoverlap = radius[i2];  // max overlap is diameter - but position is in the middle of atom
    if ((x[i2][0]<(domain->boxlo[0]-maxoverlap)) && (domain->xperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at xmin\n");
#     endif
      continue;
    } else if ((x[i2][0]>(domain->boxhi[0]+maxoverlap)) && (domain->xperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at xmax\n");
#     endif
      continue;
    } else if ((x[i2][1]<(domain->boxlo[1]-maxoverlap)) && (domain->yperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at ymin\n");
#     endif
      continue;
    } else if ((x[i2][1]>(domain->boxhi[1]+maxoverlap)) && (domain->yperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at ymax\n");
#     endif
      continue;
    } else if ((x[i2][2]<(domain->boxlo[2]-maxoverlap)) && (domain->zperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at zmin\n");
#     endif
      continue;
    } else if ((x[i2][2]>(domain->boxhi[2]+maxoverlap)) && (domain->zperiodic == 0)) {
      bondlist[n][3]=1;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen,"broken bond %d at step %ld\n",n,update->ntimestep);
        fprintf(screen, "   bond overlaped domain border at zmax\n");
#     endif
      continue;
    }

    type = bondlist[n][2]; // Get current bond type properties      

    rout= ro[type]*fmin(radius[i1],radius[i2]);

    A = M_PI * (rout*rout); // Bond Area
    

    Ainv = 1.0/A;
    Ginv = 1.0/G[type];

    J = 0.5*M_PI*(rout*rout*rout*rout);
    I  = 0.25*M_PI*(rout*rout*rout*rout);

    delx = x[i2][0] - x[i1][0]; // x-directional seperation
    dely = x[i2][1] - x[i1][1]; // y-directional seperation
    delz = x[i2][2] - x[i1][2]; // z-directional seperation 
    domain->minimum_image(delx,dely,delz); // Not 100% sure what this is... 

    rsq = delx*delx + dely*dely + delz*delz; 
    rsqinv = 1.0/rsq;
    r = sqrt(rsq);
    rinv = 1.0/r;
    r2 = r*r;
    r3 = r2*r;
    r2inv = 1.0/r2;
    r3inv = 1.0/r3;
    
    if (isnan(rinv)){
      fprintf(screen,"isnan rinv = %g\n",rinv);
      fprintf(screen,"Bond %d  atom %d and %d\n",n,i1,i2);
      error->all(FLERR,"Fue rinv");
    } 

    if (r < 1.0e-10) {
      fprintf(screen,"r = %g\n", r);
      error->all(FLERR,"distance between atoms too small\n");
    }

    // Check if bond just formed and set eq distance
    if (bondhistlist[n][12] == 0.0) {
      bondhistlist[n][12] = r;
      bondhistlist[n][13] = A;
#     ifdef LIGGGHTS_DEBUG
        fprintf(screen, "INFO: Setting bond length between %i and %i at %g\n", i1, i2, bondhistlist[n][12]);
#     endif
    }

    // Set Bond Length
    bondLength = fabs(bondhistlist[n][12]);
    if (bondLength < 1.0e-10) {
      fprintf(screen,"BondLength = %g\n", bondLength);
      error->all(FLERR,"bondlength too small\n");
    }
    
    bondLength2 = bondLength*bondLength;    
    bondLength3 = bondLength2*bondLength;
    bondLengthinv = 1.0/bondLength;
    bondLengthinv2 = 1.0/bondLength2;
    bondLengthinv3 = 1.0/bondLength3;
    phi = 120*Y[type]*I*Ainv*Ginv*r2inv/9; //bondLengthinv2
    phi_pinv = 1./(1 + phi);

    
    if (strength_flag)  //Normal distribution of strength 
    {
      fprintf(screen,"\n*******************\nInicio resistencias\n*******************\n\n\n");
      
      broken_compression = 0;
      broken_shear = 0;
      broken_tensile = 0;
      broken_total = 0;

      strength_flag = false;
      int type1,type2;

      int n2 = 3*nbondlist;
      int n3 = nbondlist;
      
      if (nbondlist%2 != 0)
      {
        n3 += 1;
      }
      
      random_number = new double[n2];
      N = new double[n3];
      Strength_c = new double[n3];
      Strength_t = new double[n3];
      Strength_s = new double[n3];
      srand(time(0));  
        
      for ( int l = 0; l < n2; l++) {
        random_number[l] = rand() / static_cast<double>(RAND_MAX);
      }
    
      int j=0;
      for (int i = 0; i < n3 / 2; i++) {
        if (i == (nbondlist + 1)/2-1)
        {
          type1 = bondlist[2*i][2];
          type2 = bondlist[2*i][2];
        } else {
          type1 = bondlist[2*i][2];
          type2 = bondlist[2*i+1][2];          
        }        
        
        N[2 * i] = sqrt(-2.0 * log(random_number[2 * j])) * sin(2 * M_PI * random_number[2 * j + 1]);
        N[2 * i + 1] = sqrt(-2.0 * log(random_number[2 * j])) * cos(2 * M_PI * random_number[2 * j + 1]);
        Strength_c[2 * i] = compression_break[type1]*(N[2*i]*CoV_compression[type1]+1);
        Strength_c[2 * i + 1] = compression_break[type2]*(N[2*i+1]*CoV_compression[type2]+1);
        Strength_t[2 * i] = tensile_break[type1]*(N[2*i]*CoV_tensile[type1]+1);
        Strength_t[2 * i + 1] = tensile_break[type2]*(N[2*i+1]*CoV_tensile[type2]+1);
        Strength_s[2 * i] = shear_break[type1]*(N[2*i]*CoV_shear[type1]+1);
        Strength_s[2 * i + 1] = shear_break[type2]*(N[2*i+1]*CoV_shear[type2]+1);
        while ((Strength_c[2 * i ] < 0) || (Strength_c[2 * i] > 2*compression_break[type1]) || isnan(Strength_c[2*i])|| (Strength_c[2 * i + 1] < 0) || (Strength_c[2 * i + 1] > 2*compression_break[type2]) || isnan(Strength_c[2*i+1])
              || (Strength_t[2 * i ] < 0) || (Strength_t[2 * i] > 2*tensile_break[type1]) || (Strength_t[2 * i + 1] < 0) || (Strength_t[2 * i + 1] > 2*tensile_break[type2])
              || (Strength_s[2 * i ] < 0) || (Strength_s[2 * i] > 2*shear_break[type1]) || (Strength_s[2 * i + 1] < 0) || (Strength_s[2 * i + 1] > 2*shear_break[type2])        ){
          j=j+1;
          N[2 * i] = sqrt(-2.0 * log(random_number[2 * j])) * sin(2 * M_PI * random_number[2 * j + 1]);
          N[2 * i + 1] = sqrt(-2.0 * log(random_number[2 * j])) * cos(2 * M_PI * random_number[2 * j + 1]);
          Strength_c[2 * i] = compression_break[type1]*(N[2*i]*CoV_compression[type1]+1);
          Strength_c[2 * i + 1] = compression_break[type2]*(N[2*i+1]*CoV_compression[type2]+1);
          Strength_t[2 * i] = tensile_break[type1]*(N[2*i]*CoV_tensile[type1]+1);
          Strength_t[2 * i + 1] = tensile_break[type2]*(N[2*i+1]*CoV_tensile[type2]+1);
          Strength_s[2 * i] = shear_break[type1]*(N[2*i]*CoV_shear[type1]+1);
          Strength_s[2 * i + 1] = shear_break[type2]*(N[2*i+1]*CoV_shear[type2]+1);
        }
        j=j+1;  

      }
      
      for (int l = 0; l < nbondlist; l++)
      {
        bondhistlist[l][ 0] = 0.0;
        bondhistlist[l][ 1] = 0.0;
        bondhistlist[l][ 2] = 0.0;
        bondhistlist[l][ 3] = 0.0;
        bondhistlist[l][ 4] = 0.0;
        bondhistlist[l][ 5] = 0.0;
        bondhistlist[l][ 6] = 0.0;
        bondhistlist[l][ 7] = 0.0;
        bondhistlist[l][ 8] = 0.0;
        bondhistlist[l][ 9] = 0.0;
        bondhistlist[l][10] = 0.0;
        bondhistlist[l][11] = 0.0;
        bondhistlist[l][20] = Strength_c[l];
        bondhistlist[l][21] = Strength_t[l];
        bondhistlist[l][22] = Strength_s[l];
      }
        
    }
    compression_strength = fabs(bondhistlist[n][20]);    
    tensile_strength = fabs(bondhistlist[n][21]);
    shear_strength = fabs(bondhistlist[n][22]);
    
    
    bondhistlist[n][20] = compression_strength;
    bondhistlist[n][21] = tensile_strength;
    bondhistlist[n][22] = shear_strength;
    

    // Transfomation matrix
    
    double tol = 1.0e-6;
    T[0][0] = delx*rinv;    
    T[0][1] = dely*rinv;
    T[0][2] = delz*rinv;
    if (fabs(T[0][0]) <= tol && fabs(T[0][1]) <= tol && (fabs(T[0][2]-1) <= tol || fabs(T[0][2]+1) <= tol))
    {
      T[1][0] = 0;    
      T[1][1] = 1;
      T[1][2] = 0;
      T[2][0] = -1;
      T[2][1] = 0;
      T[2][2] = 0;
    }else{
      T[1][0] = Z[1]*T[0][2]-Z[2]*T[0][1];    
      T[1][1] = Z[2]*T[0][0]-Z[0]*T[0][2];    
      T[1][2] = Z[0]*T[0][1]-Z[1]*T[0][0];      
      double mod_1 = sqrt(T[1][0]*T[1][0]+T[1][1]*T[1][1]+T[1][2]*T[1][2]);
      T[1][0] /= mod_1;
      T[1][1] /= mod_1;
      T[1][2] /= mod_1;
      T[2][0] = T[0][1]*T[1][2]-T[0][2]*T[1][1];
      T[2][1] = T[0][2]*T[1][0]-T[0][0]*T[1][2];
      T[2][2] = T[0][0]*T[1][1]-T[0][1]*T[1][0];
      double mod_2 = sqrt(T[2][0]*T[2][0]+T[2][1]*T[2][1]+T[2][2]*T[2][2]);
      T[2][0] /= mod_2;
      T[2][1] /= mod_2;
      T[2][2] /= mod_2;
    }
    if (isnan(T[0][0])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[0][0]");
    }
    if (isnan(T[0][1])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[0][1]");
    }
    if (isnan(T[0][2])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[0][2]");
    }
    if (isnan(T[1][0])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[1][0]");
    }
    if (isnan(T[1][1])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[1][1]");
    }
    if (isnan(T[1][2])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[1][2]");
    }
    if (isnan(T[2][0])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[2][0]");
    }
    if (isnan(T[2][1])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[2][1]");
    }
    if (isnan(T[2][2])){
      fprintf(screen,"%g    %g    %g\n",T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n",T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n",T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"delx   dely   delz\n");
      fprintf(screen,"%g    %g    %g\n",delx,dely,delz);
      fprintf(screen,"rinv =  %g\n",rinv);
      error->all(FLERR,"Fue T[2][2]");
    } /*
    if (isnan(T[0][0])) fprintf(screen,"isnan T[0][0] = %g\n",T[0][0]);
    if (isnan(T[0][1])) fprintf(screen,"isnan T[0][1] = %g\n",T[0][1]);
    if (isnan(T[0][2])) fprintf(screen,"isnan T[0][2] = %g\n",T[0][2]);
    if (isnan(T[1][0])) fprintf(screen,"isnan T[1][0] = %g\n",T[1][0]);
    if (isnan(T[1][1])) fprintf(screen,"isnan T[1][1] = %g\n",T[1][1]);
    if (isnan(T[1][2])) fprintf(screen,"isnan T[1][2] = %g\n",T[1][2]);
    if (isnan(T[2][0])) fprintf(screen,"isnan T[2][0] = %g\n",T[2][0]);
    if (isnan(T[2][1])) fprintf(screen,"isnan T[2][1] = %g\n",T[2][1]);
    if (isnan(T[2][2])) fprintf(screen,"isnan T[2][2] = %g\n",T[2][2]);
    */

    // Global displacement vector

    ug[ 0] = v[i1][0]*dt;
    ug[ 1] = v[i1][1]*dt;
    ug[ 2] = v[i1][2]*dt;
    ug[ 3] = omega[i1][0]*dt;
    ug[ 4] = omega[i1][1]*dt;
    ug[ 5] = omega[i1][2]*dt;
    ug[ 6] = v[i2][0]*dt;
    ug[ 7] = v[i2][1]*dt;
    ug[ 8] = v[i2][2]*dt;
    ug[ 9] = omega[i2][0]*dt;
    ug[10] = omega[i2][1]*dt;
    ug[11] = omega[i2][2]*dt;


    // Local displacement vector

    u[ 0] = ug[ 0]*T[0][0] + ug[ 1]*T[0][1] + ug[ 2]*T[0][2];
    u[ 1] = ug[ 0]*T[1][0] + ug[ 1]*T[1][1] + ug[ 2]*T[1][2];
    u[ 2] = ug[ 0]*T[2][0] + ug[ 1]*T[2][1] + ug[ 2]*T[2][2];
    u[ 3] = ug[ 3]*T[0][0] + ug[ 4]*T[0][1] + ug[ 5]*T[0][2];
    u[ 4] = ug[ 3]*T[1][0] + ug[ 4]*T[1][1] + ug[ 5]*T[1][2];
    u[ 5] = ug[ 3]*T[2][0] + ug[ 4]*T[2][1] + ug[ 5]*T[2][2];
    u[ 6] = ug[ 6]*T[0][0] + ug[ 7]*T[0][1] + ug[ 8]*T[0][2];
    u[ 7] = ug[ 6]*T[1][0] + ug[ 7]*T[1][1] + ug[ 8]*T[1][2];
    u[ 8] = ug[ 6]*T[2][0] + ug[ 7]*T[2][1] + ug[ 8]*T[2][2];
    u[ 9] = ug[ 9]*T[0][0] + ug[10]*T[0][1] + ug[11]*T[0][2];
    u[10] = ug[ 9]*T[1][0] + ug[10]*T[1][1] + ug[11]*T[1][2];
    u[11] = ug[ 9]*T[2][0] + ug[10]*T[2][1] + ug[11]*T[2][2];


    // Stiffness Matrix

    K[0][0] = K[6][6] = Y[type]*A*rinv; //bondLengthinv;
    K[1][1] = K[2][2] = K[7][7] = K[8][8] = 12*Y[type]*I*phi_pinv*r3inv; //bondLengthinv3;
    K[3][3] = K[9][9] = G[type]*J*rinv; //bondLengthinv;
    K[4][4] = K[5][5] = K[10][10] = K[11][11] = (4 + phi)*Y[type]*I*phi_pinv*rinv; //bondLengthinv;
    K[4][2] = K[10][2] = K[5][7] = K[11][7] = -6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[2][4] = K[2][10] = K[7][5] = K[7][11] = -6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[10][8] = K[5][1] = K[11][1] = K[8][4] = 6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[8][10] = K[1][5] = K[1][11] = K[4][8] = 6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[6][0] = K[0][6] = -Y[type]*A*rinv; //bondLengthinv;
    K[7][1] = K[8][2] = K[1][7] = K[2][8] = -12*Y[type]*I*phi_pinv*r3inv; //bondLengthinv3;
    K[9][3] = K[3][9] = -G[type]*J*rinv; //bondLengthinv;
    K[10][4] = K[11][5] = K[5][11] = K[4][10] = (2-phi)*Y[type]*I*phi_pinv*rinv; //bondLengthinv;


    // Get delta Local Forces

    /*for (int i = 0; i < 12; i++)
    {
      dF_local[i] = 0.0;
      for (int j = 0; j < 12; j++)
      {
        dF_local[i] += K[i][j]*u[j];
      }      
    }*/
    

    dF_local[ 0] = K[ 0][0]*u[0] + K[ 0][6]*u[6];
    dF_local[ 1] = K[ 1][1]*u[1] + K[ 1][5]*u[5] + K[1][7]*u[7] + K[1][11]*u[11];
    dF_local[ 2] = K[ 2][2]*u[2] + K[ 2][4]*u[4] + K[2][8]*u[8] + K[2][10]*u[10];
    dF_local[ 3] = K[ 3][3]*u[3] + K[ 3][9]*u[9];
    dF_local[ 4] = K[ 4][2]*u[2] + K[ 4][4]*u[4] + K[4][8]*u[8] + K[4][10]*u[10];
    dF_local[ 5] = K[ 5][1]*u[1] + K[ 5][5]*u[5] + K[5][7]*u[7] + K[5][11]*u[11];
    dF_local[ 6] = K[ 6][0]*u[0] + K[ 6][6]*u[6];
    dF_local[ 7] = K[ 7][1]*u[1] + K[ 7][5]*u[5] + K[7][7]*u[7] + K[7][11]*u[11];
    dF_local[ 8] = K[ 8][2]*u[2] + K[ 8][4]*u[4] + K[8][8]*u[8] + K[8][10]*u[10];
    dF_local[ 9] = K[ 9][3]*u[3] + K[ 9][9]*u[9];
    dF_local[10] = K[10][2]*u[2] + K[10][4]*u[4] + K[10][8]*u[8] + K[10][10]*u[10];
    dF_local[11] = K[11][1]*u[1] + K[11][5]*u[5] + K[11][7]*u[7] + K[11][11]*u[11];


    //increment normal and tangential force and torque 

    bondhistlist[n][ 0] += dF_local[ 0];
    bondhistlist[n][ 1] += dF_local[ 1];
    bondhistlist[n][ 2] += dF_local[ 2];
    bondhistlist[n][ 3] += dF_local[ 3];
    bondhistlist[n][ 4] += dF_local[ 4];
    bondhistlist[n][ 5] += dF_local[ 5];
    bondhistlist[n][ 6] += dF_local[ 6];
    bondhistlist[n][ 7] += dF_local[ 7];
    bondhistlist[n][ 8] += dF_local[ 8];
    bondhistlist[n][ 9] += dF_local[ 9];
    bondhistlist[n][10] += dF_local[10];
    bondhistlist[n][11] += dF_local[11];


    // Transformation matrix inv

    det_T = T[0][0]*T[1][1]*T[2][2] + T[0][1]*T[1][2]*T[2][0] + T[0][2]*T[1][0]*T[2][1] - T[0][2]*T[1][1]*T[2][0] - T[0][1]*T[1][0]*T[2][2] - T[0][0]*T[1][2]*T[2][1];
    det_Tinv = 1.0/det_T;
       
    
    Tinv[0][0] = det_Tinv*(T[1][1]*T[2][2] - T[1][2]*T[2][1]);
    Tinv[0][1] = det_Tinv*(T[0][2]*T[2][1] - T[0][1]*T[2][2]);
    Tinv[0][2] = det_Tinv*(T[0][1]*T[1][2] - T[0][2]*T[1][1]);
    Tinv[1][0] = det_Tinv*(T[1][2]*T[2][0] - T[1][0]*T[2][2]);
    Tinv[1][1] = det_Tinv*(T[0][0]*T[2][2] - T[0][2]*T[2][0]);
    Tinv[1][2] = det_Tinv*(T[0][2]*T[1][0] - T[0][0]*T[1][2]);
    Tinv[2][0] = det_Tinv*(T[1][0]*T[2][1] - T[2][0]*T[1][1]);
    Tinv[2][1] = det_Tinv*(T[0][1]*T[2][0] - T[0][0]*T[2][1]);
    Tinv[2][2] = det_Tinv*(T[0][0]*T[1][1] - T[0][1]*T[1][0]);
    if (det_T < 1.0e-10) {
      fprintf(screen,"det_T = %g\n", det_T);
      fprintf(screen,"det_Tinv = %g\n", det_Tinv);
      fprintf(screen,"%g    %g    %g\n", T[0][0],T[0][1],T[0][2]);
      fprintf(screen,"%g    %g    %g\n", T[1][0],T[1][1],T[1][2]);
      fprintf(screen,"%g    %g    %g\n", T[2][0],T[2][1],T[2][2]);
      fprintf(screen,"%g    %g    %g\n", Tinv[0][0],Tinv[0][1],Tinv[0][2]);
      fprintf(screen,"%g    %g    %g\n", Tinv[1][0],Tinv[1][1],Tinv[1][2]);
      fprintf(screen,"%g    %g    %g\n", Tinv[2][0],Tinv[2][1],Tinv[2][2]);
      error->all(FLERR,"det_T too small\n");
    }

    // Global force

    F[ 0] = bondhistlist[n][0]*Tinv[0][0] + bondhistlist[n][ 1]*Tinv[0][1] + bondhistlist[n][ 2]*Tinv[0][2];
    F[ 1] = bondhistlist[n][0]*Tinv[1][0] + bondhistlist[n][ 1]*Tinv[1][1] + bondhistlist[n][ 2]*Tinv[1][2];
    F[ 2] = bondhistlist[n][0]*Tinv[2][0] + bondhistlist[n][ 1]*Tinv[2][1] + bondhistlist[n][ 2]*Tinv[2][2];
    F[ 3] = bondhistlist[n][3]*Tinv[0][0] + bondhistlist[n][ 4]*Tinv[0][1] + bondhistlist[n][ 5]*Tinv[0][2];
    F[ 4] = bondhistlist[n][3]*Tinv[1][0] + bondhistlist[n][ 4]*Tinv[1][1] + bondhistlist[n][ 5]*Tinv[1][2];
    F[ 5] = bondhistlist[n][3]*Tinv[2][0] + bondhistlist[n][ 4]*Tinv[2][1] + bondhistlist[n][ 5]*Tinv[2][2];
    F[ 6] = bondhistlist[n][6]*Tinv[0][0] + bondhistlist[n][ 7]*Tinv[0][1] + bondhistlist[n][ 8]*Tinv[0][2];
    F[ 7] = bondhistlist[n][6]*Tinv[1][0] + bondhistlist[n][ 7]*Tinv[1][1] + bondhistlist[n][ 8]*Tinv[1][2];
    F[ 8] = bondhistlist[n][6]*Tinv[2][0] + bondhistlist[n][ 7]*Tinv[2][1] + bondhistlist[n][ 8]*Tinv[2][2];
    F[ 9] = bondhistlist[n][9]*Tinv[0][0] + bondhistlist[n][10]*Tinv[0][1] + bondhistlist[n][11]*Tinv[0][2];
    F[10] = bondhistlist[n][9]*Tinv[1][0] + bondhistlist[n][10]*Tinv[1][1] + bondhistlist[n][11]*Tinv[1][2];
    F[11] = bondhistlist[n][9]*Tinv[2][0] + bondhistlist[n][10]*Tinv[2][1] + bondhistlist[n][11]*Tinv[2][2];
    
    
    
    //flag breaking of bond if criterion met
    
    comp_a = bondhistlist[n][6]*Ainv - rout*sqrt(bondhistlist[n][4]*bondhistlist[n][4] + bondhistlist[n][5]*bondhistlist[n][5])/I;
    comp_b = bondhistlist[n][6]*Ainv - rout*sqrt(bondhistlist[n][10]*bondhistlist[n][10] + bondhistlist[n][11]*bondhistlist[n][11])/I;
    comp_stress = -MIN(comp_a,comp_b);
    tens_a = bondhistlist[n][6]*Ainv + rout*sqrt(bondhistlist[n][4]*bondhistlist[n][4] + bondhistlist[n][5]*bondhistlist[n][5])/I;
    tens_b = bondhistlist[n][6]*Ainv + rout*sqrt(bondhistlist[n][10]*bondhistlist[n][10] + bondhistlist[n][11]*bondhistlist[n][11])/I;
    tens_stress = MAX(tens_a,tens_b);
    shear_stress = fabs(bondhistlist[n][3])*rout/(2*I) + 4*sqrt(bondhistlist[n][1]*bondhistlist[n][1] + bondhistlist[n][2]*bondhistlist[n][2])*Ainv/3;
        
        
    bondhistlist[n][14] = comp_stress;
    bondhistlist[n][15] = tens_stress;
    bondhistlist[n][16] = shear_stress;


    comp_break = compression_strength < comp_stress;
    tens_break = tensile_strength < tens_stress;
    tau_break = shear_strength < shear_stress;

    
    if(comp_break || tens_break || tau_break)
    {
      bondlist[n][3] = 1;
      bondhistlist[n][17] = 1.0;
      bondhistlist[n][18] = 1.0;
      bondhistlist[n][19] = 1.0;
      broken_total += 1;

      if(comp_break)
      {
        broken_compression += 1;
        //fprintf(screen, "   it was compression stress\n");
        //fprintf(screen,"   compression_break == %e\n      mag_force == %e\n",compression_strength,comp_stress);
      }
      if(tens_break)
      {
        broken_tensile += 1;
        //fprintf(screen, "   it was tensile stress\n");
        //fprintf(screen,"   tensile_break == %e\n      mag_force == %e\n",tensile_strength,tens_stress);
      }
      if(tau_break)
      {
        broken_shear += 1;
        //fprintf(screen,"   it was shear stress\n");
        //fprintf(screen,"   shear_break == %e\n      mag_force == %e\n",shear_strength,shear_stress);
      }
      continue;
    }else
    {    
      bondhistlist[n][17] = comp_stress/compression_strength;
      bondhistlist[n][18] = tens_stress/tensile_strength;
      bondhistlist[n][19] = shear_stress/shear_strength;
    }
    
    
        
    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      
      f[i1][0] += -F[0];
      f[i1][1] += -F[1];
      f[i1][2] += -F[2];

      torque[i1][0] += -F[3];
      torque[i1][1] += -F[4];
      torque[i1][2] += -F[5];
      
    }

    if (newton_bond || i2 < nlocal) {
      
      f[i2][0] += -F[6];
      f[i2][1] += -F[7];
      f[i2][2] += -F[8];

      torque[i2][0] += -F[9];
      torque[i2][1] += -F[10];
      torque[i2][2] += -F[11];
      
    }
  
  }
  for (int n = 0; n < nbondlist; n++){
    if (bondlist[n][3] == 1){
      i1 = bondlist[n][0]; // Sphere 1
      i2 = bondlist[n][1]; // Sphere 2
      if (i1 < nlocal) {
        for (int k1 = 0; k1 < atom->num_bond[i1]; k1++)
        {
          int j2 = atom->map(atom->bond_atom[i1][k1]); //mapped index of bond-partner
          if (i2 == j2)
          {
              atom->bond_type[i1][k1] = 0;
              break;
          }
        }
      }
      if (i2 < nlocal) {
        for (int k1 = 0; k1 < atom->num_bond[i2]; k1++)
        {
          int j1 = atom->map(atom->bond_atom[i2][k1]); //mapped index of bond-partner
          if (i1 == j1)
          {
              atom->bond_type[i2][k1] = 0;
              break;
          }
        }
      }
    }
  }
  


  /*//Copy bond status to bond_intact
  int **bond_atom = atom->bond_atom;
  int **bond_intact = atom->bond_intact;
  int neighID,atom1,atom2;

  for (int i = 0; i < nlocal; i++) {
    atom1 = i;
    for (int m = 0; m < atom->num_bond[atom1]; m++) {
      atom2 = atom->map(bond_atom[atom1][m]);
      if (newton_bond || i < atom2) {
        neighID = -1;
        for (int k = 0; k < nbondlist; k++) {
          if ((bondlist[k][0]==atom1 && bondlist[k][1]==atom2) || ((bondlist[k][1]==atom1 && bondlist[k][0]==atom2))) {
            neighID = k;
            if (bondlist[neighID][3])
            {
              bond_intact[i][m] = 0;
            }
            break;
          }
        }
      }
    }
  }*/
}

/* ---------------------------------------------------------------------- */

void BondTBBM::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;
  if (strength_flag == false)
  {
    strength_flag = true;
  }
  
  
  // Create bond property variables
  memory->create(ro,n+1,"bond:ro");
  memory->create(Y,n+1,"bond:Y");
  memory->create(G,n+1,"bond:G");
  
  // Create bond break variables
  memory->create(compression_break,n+1,"bond:compression_break");
  memory->create(shear_break,n+1,"bond:shear_break");
  memory->create(tensile_break,n+1,"bond:tensile_break");
  memory->create(CoV_compression,n+1,"bond:CoV_compression");
  memory->create(CoV_tensile,n+1,"bond:CoV_tensile");
  memory->create(CoV_shear,n+1,"bond:CoV_shear");

  /*broken_total = 0;
  broken_compression = 0;
  broken_shear = 0;
  broken_tensile = 0;*/
  

  memory->create(setflag,(n+1),"bond:setflag");
  for (int i = 1; i <= n; i++)
    setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondTBBM::coeff(int narg, char **arg)
{

  if (narg != 10){
    error->all(FLERR,"Incorrect args for bond coefficients");
  }

  strength_flag = true;
  

  int arg_id = 1;
  double ro_one = force->numeric(FLERR,arg[arg_id++]);
  double Y_one = force->numeric(FLERR,arg[arg_id++]);
  double G_one = force->numeric(FLERR,arg[arg_id]);
  
  
  if (ro_one <= 0){
    error->all(FLERR,"ro must be greater than 0");
  }
  
    

  if (!allocated)
    allocate();

  double compression_break_one = force->numeric(FLERR,arg[++arg_id]);
  double CoV_compression_one = force->numeric(FLERR,arg[++arg_id]);
  double tensile_break_one = force->numeric(FLERR,arg[++arg_id]);
  double CoV_tensile_one = force->numeric(FLERR,arg[++arg_id]);
  double shear_break_one = force->numeric(FLERR,arg[++arg_id]);
  double CoV_shear_one = force->numeric(FLERR,arg[++arg_id]);

  

  if(comm->me == 0) {
    fprintf(screen,"\n--- Bond Parameters Being Set ---\n");
    fprintf(screen,"   ro == %g\n",ro_one);
    fprintf(screen,"   Y == %g\n",Y_one);
    fprintf(screen,"   G == %g\n",G_one);
    fprintf(screen,"   compression_break == %g\n",compression_break_one);
    fprintf(screen,"   CoV_compression == %g\n",CoV_compression_one);
    fprintf(screen,"   tensile_break == %g\n",tensile_break_one);
    fprintf(screen,"   CoV_tensile == %g\n",CoV_tensile_one);
    fprintf(screen,"   shear_break == %g\n",shear_break_one);
    fprintf(screen,"   CoV_shear == %g\n",CoV_shear_one);    
    fprintf(screen,"--- End Bond Parameters ---\n\n");
  }

  int ilo, ihi;
  force->bounds(arg[0],atom->nbondtypes,ilo,ihi);
  int count = 0;
  for (int i = ilo; i <= ihi; ++i) {
    ro[i] = ro_one;
    Y[i] = Y_one;
    G[i] = G_one;

    compression_break[i] = compression_break_one;
    CoV_compression[i] = CoV_compression_one;
    tensile_break[i] = tensile_break_one;
    CoV_tensile[i] = CoV_tensile_one;
    shear_break[i] = shear_break_one;
    CoV_shear[i] = CoV_shear_one;

    if (CoV_compression[i] < 0 || CoV_compression[i] > 1) {
      error->all(FLERR,"The range of values for CoV are zero to one");
    }
    if (CoV_tensile[i] < 0 || CoV_tensile[i] > 1) {
      error->all(FLERR,"The range of values for CoV are zero to one");
    }
    if (CoV_shear[i] < 0 || CoV_shear[i] > 1) {
      error->all(FLERR,"The range of values for CoV are zero to one");
    }
    
    setflag[i] = 1;
    ++count;
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for bond coefficients - or the bonds are not initialized in create_atoms");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondTBBM::equilibrium_distance(int i)
{
  //NP ATTENTION: this is _not_ correct - and has implications on fix shake, pair_lj_cut_coul_long and pppm
  //NP it is not possible to define a general equilibrium distance for this bond model
  //NP as rotational degree of freedom is present
  return 0.;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondTBBM::write_restart(FILE *fp)
{
  fwrite(&ro[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&Y[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&G[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&compression_break[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&CoV_compression[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&tensile_break[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&CoV_tensile[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&shear_break[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&CoV_shear[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondTBBM::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&ro[1],sizeof(double),atom->nbondtypes,fp);
    fread(&Y[1],sizeof(double),atom->nbondtypes,fp);
    fread(&G[1],sizeof(double),atom->nbondtypes,fp);
    fread(&compression_break[1],sizeof(double),atom->nbondtypes,fp);
    fread(&CoV_compression[1],sizeof(double),atom->nbondtypes,fp);
    fread(&tensile_break[1],sizeof(double),atom->nbondtypes,fp);
    fread(&CoV_tensile[1],sizeof(double),atom->nbondtypes,fp);
    fread(&shear_break[1],sizeof(double),atom->nbondtypes,fp);
    fread(&CoV_shear[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&ro[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&Y[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&G[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&compression_break[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&CoV_compression[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&tensile_break[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&CoV_tensile[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&shear_break[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&CoV_shear[1],atom->nbondtypes,MPI_DOUBLE,0,world);


  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}


/* ---------------------------------------------------------------------- */

double BondTBBM::single(int type, double rsq, int i, int j, double &fforce)
{
  error->all(FLERR,"Bond granular does not support this feature");
  return 0.0;
}

// Use method given by Guo et al. (2013) to determine stable time step for a bonded particle
double BondTBBM::getMinDt()
{
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;

  double curDt = 0.0;
  double minDt = 1.0;
  double *radius = atom->radius;
  double *density = atom->density;
  double Kmax;
  double K[3][3];
  double rinv,r2inv,r3inv,Ainv,Ginv,I,A,J;
  double r,phi,phi_pinv,rout;
  double **x = atom->x;
  double m1,m2,m_min;
  int i1,i2,type;

  for (int k = 0; k < nbondlist; k++) {
    if (bondlist[k][3]) continue;
    i1 = bondlist[k][0];
    i2 = bondlist[k][1];
    type = bondlist[k][2];
    r = sqrt((x[i2][0]-x[i1][0])*(x[i2][0]-x[i1][0])+(x[i2][1]-x[i1][1])*(x[i2][1]-x[i1][1])+(x[i2][2]-x[i1][2]*(x[i2][2]-x[i1][2])));
    rinv = 1.0/r;
    r2inv = rinv/r;
    r3inv = r2inv/r;

    rout= ro[type]*fmin(radius[i1],radius[i2]);
    A = M_PI * (rout*rout);
    I = M_PI*0.25*(rout*rout*rout*rout);
    J = M_PI*0.5*(rout*rout*rout*rout);
    Ainv = 1.0/A;
    Ginv = 1.0/G[type];
    m1 = 4.1887902047863909846168578443*density[i1]*radius[i1]*radius[i1]*radius[i1];
    m2 = 4.1887902047863909846168578443*density[i2]*radius[i2]*radius[i2]*radius[i2];
    m_min = MIN(m1,m2);

    
    phi = 120*Y[type]*I*Ainv*Ginv*r2inv/9;
    phi_pinv = 1./(1 + phi);
    
    K[0][0] = K[6][6] = Y[type]*A*rinv; //bondLengthinv;
    K[1][1] = K[2][2] = K[7][7] = K[8][8] = 12*Y[type]*I*phi_pinv*r3inv; //bondLengthinv3;
    K[3][3] = K[9][9] = G[type]*J*rinv; //bondLengthinv;
    K[4][4] = K[5][5] = K[10][10] = K[11][11] = (4 + phi)*Y[type]*I*phi_pinv*rinv; //bondLengthinv;
    K[4][2] = K[10][2] = K[5][7] = K[11][7] = -6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[2][4] = K[2][10] = K[7][5] = K[7][11] = -6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[10][8] = K[5][1] = K[11][1] = K[8][4] = 6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[8][10] = K[1][5] = K[1][11] = K[4][8] = 6*Y[type]*I*phi_pinv*r2inv; //bondLengthinv2;
    K[6][0] = K[0][6] = -Y[type]*A*rinv; //bondLengthinv;
    K[7][1] = K[8][2] = K[1][7] = K[2][8] = -12*Y[type]*I*phi_pinv*r3inv; //bondLengthinv3;
    K[9][3] = K[3][9] = -G[type]*J*rinv; //bondLengthinv;
    K[10][4] = K[11][5] = K[5][11] = K[4][10] = (2-phi)*Y[type]*I*phi_pinv*rinv; //bondLengthinv;
    
    for (int i=0; i<3; i++){
      for (int j = 0; j < 3; j++)
      {
        if (Kmax<K[i][j])
        {
          Kmax = K[i][j];
        }        
      }      
    }    
    curDt = 2*sqrt(m_min/Kmax);
    if (curDt < minDt) minDt = curDt;
  }
  
  fprintf(screen,"minDt = %f\n",minDt);
  return minDt;
}
