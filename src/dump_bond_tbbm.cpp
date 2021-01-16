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

#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "dump_bond_tbbm.h"
#include "atom.h"
#include "modify.h"
#include "fix.h"
#include "compute.h"
#include "domain.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;

enum{INT,DOUBLE};

#define ONEFIELD 32
#define DELTA 1048576

/* ---------------------------------------------------------------------- */

DumpBondTBBM::DumpBondTBBM(LAMMPS *lmp, int narg, char **arg) :
  Dump(lmp, narg, arg)
{
  if (narg == 5) error->all(FLERR,"No dump local arguments specified");

  clearstep = 1;

  nevery = force->inumeric(FLERR,arg[3]);

  size_one = nfield = narg-5;
  pack_choice = new FnPtrPack[nfield];
  vtype = new int[nfield];

  arg_pos = new int[nfield];
  nbonds = count_bonds(0);
  indices1 = new int[nbonds];
  indices2 = new int[nbonds];

  memory->create(array,nbonds,nfield,"bond/gran:array");
  buf = &array[0][0];
  nbonds = count_bonds(1);

  buffer_allow = 1;
  buffer_flag = 1;

  // computes & fixes which the dump accesses


  // process attributes

  parse_fields(narg,arg);

  // setup format strings

  vformat = new char*[size_one];

  format_default = new char[3*size_one+1];
  format_default[0] = '\0';

  for (int i = 0; i < size_one; i++) {
    if (vtype[i] == INT) format_default = strcat(format_default,"%d ");
    else format_default = strcat(format_default,"%g ");
    vformat[i] = NULL;
  }

}

/* ---------------------------------------------------------------------- */

DumpBondTBBM::~DumpBondTBBM()
{
  delete [] pack_choice;
  delete [] vtype;
  delete [] field2index;
  delete [] argindex;
  delete [] indices1;
  delete [] indices2;
  delete [] arg_pos;


  for (int i = 0; i < size_one; i++) delete [] vformat[i];
  delete [] vformat;

}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::init_style()
{
  if (sortBuffer && sortBuffer->sort_set() && sortBuffer->get_sortcol() == 0)
    error->all(FLERR,"Dump local cannot sort by atom ID");

  delete [] format;
  char *str;
  if (format_user) str = format_user;
  else str = format_default;

  int n = strlen(str) + 1;
  format = new char[n];
  strcpy(format,str);

  // tokenize the format string and add space at end of each format element

  char *ptr;
  for (int i = 0; i < size_one; i++) {
    if (i == 0) ptr = strtok(format," \0");
    else ptr = strtok(NULL," \0");
    delete [] vformat[i];
    vformat[i] = new char[strlen(ptr) + 2];
    strcpy(vformat[i],ptr);
    vformat[i] = strcat(vformat[i]," ");
  }

  // setup boundary string

  domain->boundary_string(boundstr);

  // setup function ptrs

  if (buffer_flag == 1) write_choice = &DumpBondTBBM::write_string;
  else write_choice = &DumpBondTBBM::write_lines;

  
  // open single file, one time only

  if (multifile == 0) openfile();
}

/* ---------------------------------------------------------------------- */

int DumpBondTBBM::modify_param(int narg, char **arg)
{
  return 0;
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::write_header(bigint ndump)
{
  if (me == 0) {
    fprintf(fp,"# vtk DataFile Version 4.1\n");
    fprintf(fp,"Generated by LIGGGHTS\n");
    fprintf(fp,"ASCII\n");
    fprintf(fp,"DATASET POLYDATA\n");
  }

}

/* ---------------------------------------------------------------------- */

int DumpBondTBBM::count()
{
  nmine = count_bonds(0);
  return nmine;
}

int DumpBondTBBM::count_bonds(int flag)
{

  int i,atom1,atom2;

  int *num_bond = atom->num_bond;
  int **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  double delx,dely,delz,rsq,radsum;
  double **x = atom->x;
  double *radius = atom->radius;
  int i1,i2;
  int m;

  //bigint totalbond = 0;
                     
  for (atom1 = 0; atom1 < nlocal; atom1++) {
    if (!(mask[atom1] & groupbit)) continue;
    for (i = 0; i < num_bond[atom1]; i++) {
      atom2 = atom->map(bond_atom[atom1][i]);
      if (atom2 < 0 || !(mask[atom2] & groupbit)) continue;
      if (newton_bond == 0 && tag[atom1] > tag[atom2]) continue;
      i1 = atom1;
      i2 = atom2;
      delx = x[i2][0] - x[i1][0];
      dely = x[i2][1] - x[i1][1];
      delz = x[i2][2] - x[i1][2];
      rsq = delx*delx + dely*dely + delz*delz;
      radsum = radius[i1] + radius[i2];
      if (bond_type[atom1][i] == 0 && rsq > radsum * radsum) continue;
      if (bond_type[atom1][i] < 0) continue;

      if (flag) {
        indices1[m] = atom1;
        indices2[m] = i;
      }
      m++;
    }
  }
  
  /*

  fprintf(screen,"antes bondall\n");
  bigint bondall;
  MPI_Allreduce(&totalbond,&bondall,1,MPI_LMP_BIGINT,MPI_SUM,world);
  m = bondall; // /2; //every bond is counted twice (for every partner-atom)
 */
  return m;
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack(int *dummy)
{
  delete [] indices1;
  delete [] indices2;
  nbonds = count_bonds(0);
  indices1 = new int[nbonds];
  indices2 = new int[nbonds];
  nbonds = count_bonds(1);
  for (int n = 0; n < size_one; n++) (this->*pack_choice[n])(n);
}

/* ----------------------------------------------------------------------
   convert mybuf of doubles to one big formatted string in sbuf
   return -1 if strlen exceeds an int, since used as arg in MPI calls in Dump
------------------------------------------------------------------------- */

int DumpBondTBBM::convert_string(int n, double *mybuf)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int nbonds3 = 3*nbonds;


  int j = 0;
  int k = 1;
  int param = nfield -2;
  int l;

  if (nforce == 3)
  {
    param = param - 2;
  }
  if (ntorque1 == 3)
  {
    param = param - 2;
  }
  if (ntorque2 == 3)
  {
    param = param - 2;
  }


  int offset = 0;
  int m = 0;
  double rows = 1 + ceil(nlocal/3);
  if (offset + rows*ONEFIELD > maxsbuf) {
    if ((bigint) maxsbuf + DELTA > MAXSMALLINT) return -1;
    maxsbuf += DELTA;
    memory->grow(sbuf,maxsbuf,"dump:sbuf");
  }
  offset += sprintf(&sbuf[offset],"POINTS %i float\n",nlocal);
  for (int i = 0; i < nlocal; i++) {
    if (offset + rows*ONEFIELD > maxsbuf) {
      if ((bigint) maxsbuf + DELTA > MAXSMALLINT) return -1;
      maxsbuf += DELTA;
      memory->grow(sbuf,maxsbuf,"dump:sbuf");
    }
    if (i+1 == nlocal){
      offset += sprintf(&sbuf[offset],"%g %g %g\n",x[i][0],x[i][1],x[i][2]);
      break;
    } 
    if (i+2 == nlocal) {
      offset += sprintf(&sbuf[offset],"%g %g %g %g %g %g\n",x[i][0],x[i][1],x[i][2],x[i+1][0],x[i+1][1],x[i+1][2]);
      break;
    }
    offset += sprintf(&sbuf[offset],"%g %g %g %g %g %g %g %g %g\n",x[i][0],x[i][1],x[i][2],x[i+1][0],x[i+1][1],x[i+1][2],x[i+2][0],x[i+2][1],x[i+2][2]);
    i += 2;
  }
  offset += sprintf(&sbuf[offset],"\n");
  offset += sprintf(&sbuf[offset],"LINES %d %d\n", nbonds, nbonds3);
  for (int i = 0; i < nbonds; i++) {
    if (offset + size_one*ONEFIELD > maxsbuf) {
      if ((bigint) maxsbuf + DELTA > MAXSMALLINT) return -1;
      maxsbuf += DELTA;
      memory->grow(sbuf,maxsbuf,"dump:sbuf");
    }
    offset += sprintf(&sbuf[offset],"2 %g %g\n",mybuf[j],mybuf[k]);
    j += size_one;
    k += size_one;
  }
  offset += sprintf(&sbuf[offset],"\nCELL_DATA %i\n",nbonds);
  offset += sprintf(&sbuf[offset],"FIELD FielData %d\n",param);
  for (int i = 0; i < nfield; i++) {
    if (offset + (1+nbonds)*ONEFIELD > maxsbuf) {
      if ((bigint) maxsbuf + (1+nbonds)*ONEFIELD > MAXSMALLINT) return -1;
      maxsbuf += (1+nbonds)*ONEFIELD;
      memory->grow(sbuf,maxsbuf,"dump:sbuf");
    }     
    if (arg_pos[i] == 3) { //bondtype
      k = i;
      offset += sprintf(&sbuf[offset],"bondtype 1 %i int\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 4) { //bforceN
      k = i;
      offset += sprintf(&sbuf[offset],"bondforceN 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 5) { //bforceT
      k = i;
      offset += sprintf(&sbuf[offset],"bondforceT 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 6) { //btorqueN
      k = i;
      offset += sprintf(&sbuf[offset],"bondtorqueN 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 7) { //btorqueT
      k = i;
      offset += sprintf(&sbuf[offset],"bondtorqueT 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 8) { //compression_stress
      k = i;
      offset += sprintf(&sbuf[offset],"compression_stress 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 9) { //tensile_stress
      k = i;
      offset += sprintf(&sbuf[offset],"tensile_stress 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 10) { //shear_stress
      k = i;
      offset += sprintf(&sbuf[offset],"shear_stress 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 11) { //beqdist
      k = i;
      offset += sprintf(&sbuf[offset],"bondeqdist 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 12) { //barea
      k = i;
      offset += sprintf(&sbuf[offset],"bondarea 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 13) { //bondbroken
      k = i;
      offset += sprintf(&sbuf[offset],"bondbroken 1 %i int\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 14) { //compression_ratio
      k = i;
      offset += sprintf(&sbuf[offset],"compression_ratio 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 15) { //tensile_ratio
      k = i;
      offset += sprintf(&sbuf[offset],"tensile_ratio 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 16) { //sheaar_ratio
      k = i;
      offset += sprintf(&sbuf[offset],"shear_ratio 1 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g\n",mybuf[k]);
        k += size_one;
      }
    } else if (arg_pos[i] == 17) { //bforceX-bforceY-bforceZ
      k = i + 2;
      l = i + 1;
      m = i;
      offset += sprintf(&sbuf[offset],"bforce 3 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g %g %g\n",mybuf[m],mybuf[l],mybuf[k]);
        k += size_one;
        l += size_one;
        m += size_one;
      }
    } else if (arg_pos[i] == 20) { //btorqueX1-btorqueY1-btorqueZ1
      k = i + 2;
      l = i + 1;
      m = i;
      offset += sprintf(&sbuf[offset],"btorque1 3 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g %g %g\n",mybuf[m],mybuf[l],mybuf[k]);
        k += size_one;
        l += size_one;
        m += size_one;
      }
    } else if (arg_pos[i] == 23) { //btorqueX2-btorqueY2-btorqueZ2
      k = i + 2;
      l = i + 1;
      m = i;
      offset += sprintf(&sbuf[offset],"btorque2 3 %i double\n",nbonds);
      for (int j = 0; j < nbonds; j++) {
        offset += sprintf(&sbuf[offset],"%g %g %g\n",mybuf[m],mybuf[l],mybuf[k]);
        k += size_one;
        l += size_one;
        m += size_one;
      }
    }
  }

  return offset;
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::write_data(int n, double *mybuf)
{
  (this->*write_choice)(n,mybuf);
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::write_string(int n, double *mybuf)
{
  fwrite(mybuf,sizeof(char),n,fp);
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::write_lines(int n, double *mybuf)
{
  int i,j;

  int m = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < size_one; j++) {
      if (vtype[j] == INT) fprintf(fp,vformat[j],static_cast<int> (mybuf[m]));
      else fprintf(fp,vformat[j],mybuf[m]);
      m++;
    }
    fprintf(fp,"\n");
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::parse_fields(int narg, char **arg)
{


  int i;
  nforce = 0;
  ntorque1 = 0;
  ntorque2 = 0;

  if (strcmp(arg[5],"batom1") != 0)
  {
    error->all(FLERR,"The first argument in dump bond/tbbm must be batom1");
  } else if (strcmp(arg[6],"batom2") != 0)
  {
    error->all(FLERR,"The second argument in dump bond/tbbm must be batom2");
  }


  for (int iarg = 5; iarg < narg; iarg++)
  {
    i = iarg-5;
    if (strcmp(arg[iarg],"batom1") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_batom1;   
      arg_pos[i] = 1; 
      vtype[i] = INT;
    } else if (strcmp(arg[iarg],"batom2") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_batom2;
      arg_pos[i] = 2; 
      vtype[i] = INT;
    } else if (strcmp(arg[iarg],"bondtype") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_bondtype;
      arg_pos[i] = 3; 
      vtype[i] = INT;
    } else if (strcmp(arg[iarg],"bforceN") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_bforceN;
      arg_pos[i] = 4; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"bforceT") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_bforceT;
      arg_pos[i] = 5; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"btorqueN") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueN;
      arg_pos[i] = 6; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"btorqueT") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueT;
      arg_pos[i] = 7; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"compression_stress") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_compression_stress;
      arg_pos[i] = 8; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"tensile_stress") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_tensile_stress;
      arg_pos[i] = 9; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"shear_stress") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_shear_stress;
      arg_pos[i] = 10; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"beqdist") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_beqdist;
      arg_pos[i] = 11; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"barea") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_barea;
      arg_pos[i] = 12; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"bondbroken") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_bondbroken;
      arg_pos[i] = 13; 
      vtype[i] = INT;
    } else if (strcmp(arg[iarg],"compression_ratio") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_compression_ratio;
      arg_pos[i] = 14; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"tensile_ratio") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_tensile_ratio;
      arg_pos[i] = 15; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"shear_ratio") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_shear_ratio;
      arg_pos[i] = 16; 
      vtype[i] = DOUBLE;
    } else if (strcmp(arg[iarg],"bforceX") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_bforceX;
      arg_pos[i] = 17; 
      vtype[i] = DOUBLE;
      nforce++;
    } else if (strcmp(arg[iarg],"bforceY") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_bforceY;
      arg_pos[i] = 18; 
      vtype[i] = DOUBLE;
      nforce++;
    } else if (strcmp(arg[iarg],"bforceZ") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_bforceZ;
      arg_pos[i] = 19; 
      vtype[i] = DOUBLE;
      nforce++;
    } else if (strcmp(arg[iarg],"btorqueX1") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueX1;
      arg_pos[i] = 20; 
      vtype[i] = DOUBLE;
      ntorque1++;
    } else if (strcmp(arg[iarg],"btorqueY1") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueY1;
      arg_pos[i] = 21; 
      vtype[i] = DOUBLE;
      ntorque1++;
    } else if (strcmp(arg[iarg],"btorqueZ1") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueZ1;
      arg_pos[i] = 22; 
      vtype[i] = DOUBLE;
      ntorque1++;
    } else if (strcmp(arg[iarg],"btorqueX2") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueX2;
      arg_pos[i] = 23; 
      vtype[i] = DOUBLE;
      ntorque2++;
    } else if (strcmp(arg[iarg],"btorqueY2") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueY2;
      arg_pos[i] = 24; 
      vtype[i] = DOUBLE;
      ntorque2++;
    } else if (strcmp(arg[iarg],"btorqueZ2") == 0)
    {
      pack_choice[i] = &DumpBondTBBM::pack_btorqueZ2;
      arg_pos[i] = 25; 
      vtype[i] = DOUBLE;
      ntorque2++;
    } else
    {
      error->all(FLERR,"Invalid keyword in dump bond/tbbm command"); 
    }
  }
  if (nforce != 0 && nforce != 3)
  {
    error->all(FLERR,"The use of bforceX, bforceY and bforceZ together is required");
  }
  if (ntorque1 != 0 && ntorque1 != 3)
  {
    error->all(FLERR,"The use of btorqueX1, btorqueY1 and btorqueZ1 together is required");
  }
  if (ntorque2 != 0 && ntorque2 != 3)
  {
    error->all(FLERR,"The use of btorqueX2, btorqueY2 and btorqueZ2 together is required");
  }

}

/* ----------------------------------------------------------------------
   add Compute to list of Compute objects used by dump
   return index of where this Compute is in list
   if already in list, do not add, just return index, else add to list
------------------------------------------------------------------------- */

int DumpBondTBBM::add_compute(char *id)
{
  return 0;
}

/* ----------------------------------------------------------------------
   add Fix to list of Fix objects used by dump
   return index of where this Fix is in list
   if already in list, do not add, just return index, else add to list
------------------------------------------------------------------------- */

int DumpBondTBBM::add_fix(char *id)
{
  return 0;
}

/* ----------------------------------------------------------------------
   extraction of Compute, Fix results
------------------------------------------------------------------------- */


void DumpBondTBBM::pack_batom1(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  int batom1 = 0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            batom1 = i1;
            break;
        }
     }
    buf[n] = batom1; 
    n += size_one;
  }

}

void DumpBondTBBM::pack_batom2(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  int batom2 = 0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            batom2 = j1;
            break;
        }
     }
    buf[n] = batom2; 
    n += size_one;
  }
}

void DumpBondTBBM::pack_bondbroken(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  int bbroken = 0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            bbroken = bondlist[k][3];
            break;
        }
     }
    buf[n] = bbroken; 
    n += size_one;
  }
}


void DumpBondTBBM::pack_beqdist(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double r = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            if (bondhistlist[k][12]<0)
            {
              r=-bondhistlist[k][12];
            } else
            {
              r=bondhistlist[k][12];
            }        
            break;
        }
     }
    buf[n] = r;
    n += size_one;
  }
}

void DumpBondTBBM::pack_bforceN(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double f = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            f = sqrt(bondhistlist[k][0]*bondhistlist[k][0] + bondhistlist[k][1]*bondhistlist[k][1] + bondhistlist[k][2]*bondhistlist[k][2]);
            break;
        }
     }
    buf[n] = f;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_bforceT(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double f = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            f=sqrt(bondhistlist[k][3]*bondhistlist[k][3] + bondhistlist[k][4]*bondhistlist[k][4] + bondhistlist[k][5]*bondhistlist[k][5]);
            break;
        }
     }
    buf[n] = f;
    n +=size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueN(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t=sqrt(bondhistlist[k][6]*bondhistlist[k][6] + bondhistlist[k][7]*bondhistlist[k][7] + bondhistlist[k][8]*bondhistlist[k][8]);
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueT(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t=sqrt(bondhistlist[k][9]*bondhistlist[k][9] + bondhistlist[k][10]*bondhistlist[k][10] + bondhistlist[k][11]*bondhistlist[k][11]);
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_compression_stress(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double s = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            s = bondhistlist[k][14];
            break;
        }
     }
    buf[n] = s;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_tensile_stress(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double s = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            s = bondhistlist[k][15];
            break;
        }
     }
    buf[n] = s;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_shear_stress(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double s = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;  
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            s = bondhistlist[k][16];
            break;
        }
     }
    buf[n] = s;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_barea(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double A = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            A = bondhistlist[k][13];
            break;
        }
     }
    buf[n] = A;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_bondtype(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  int bondtype = 0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            bondtype = bondlist[k][2];
            break;
        }
     }
    buf[n] = bondtype;
    n += size_one;
  } 
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_compression_ratio(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double ratio = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            ratio = bondhistlist[k][17];
            break;
        }
     }
    buf[n] = ratio;
    n += size_one;
  } 
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_tensile_ratio(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double ratio = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            ratio = bondhistlist[k][18];
            break;
        }
     }
    buf[n] = ratio;
    n += size_one;
  } 
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_shear_ratio(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double ratio = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            ratio = bondhistlist[k][19];
            break;
        }
     }
    buf[n] = ratio;
    n += size_one;
  } 
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_bforceX(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double f = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            f = bondhistlist[k][0];
            break;
        }
     }
    buf[n] = f;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_bforceY(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double f = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;
  
  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            f = bondhistlist[k][1];
            break;
        }
     }
    buf[n] = f;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_bforceZ(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double f = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            f = bondhistlist[k][2];
            break;
        }
     }
    buf[n] = f;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueX1(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t = bondhistlist[k][3];
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueY1(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t = bondhistlist[k][4];
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueZ1(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t = bondhistlist[k][5];
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueX2(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t = bondhistlist[k][9];
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueY2(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t = bondhistlist[k][10];
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpBondTBBM::pack_btorqueZ2(int n)
{
  int i,j,k,i1,j1,atom1,atom2;
  double t = 0.0;
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;

  //search for every pair i,j in bondlist
  for (int m = 0; m < nbonds; m++) {
    i = indices1[m];
    j = indices2[m];
    atom1=i;                                 //intern index of selected atom
    atom2=atom->map(atom->bond_atom[i][j]);  //mapped index of bond-partner
    if (atom2<0) continue; //is not on this proc, no Idea how to handle this ..

     for (k = 0; k < nbondlist; k++) {
        i1 = bondlist[k][0];
        j1 = bondlist[k][1];
        //printf("|-> i1=%d,j1=%d",atom1,atom2);
        if (((atom1==i1) && (atom2==j1)) || ((atom1==j1) && (atom2==i1))) {
            t = bondhistlist[k][11];
            break;
        }
     }
    buf[n] = t;
    n += size_one;
  }
}