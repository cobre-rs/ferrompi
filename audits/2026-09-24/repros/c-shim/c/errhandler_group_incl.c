#include <mpi.h>
#include <stdio.h>
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  MPI_Group g, ng; MPI_Comm_group(MPI_COMM_WORLD, &g);
  int bad = 999;
  int rc = MPI_Group_incl(g, 1, &bad, &ng);
  printf("Group_incl rc=%d (returned, not aborted)\n", rc);
  MPI_Datatype t; rc = MPI_Type_contiguous(-1, MPI_INT, &t);
  printf("Type_contiguous rc=%d\n", rc);
  MPI_Finalize(); return 0;
}
