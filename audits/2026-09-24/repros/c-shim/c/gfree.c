#include <mpi.h>
#include <stdio.h>
int main(void){ MPI_Init(NULL,NULL); MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
 MPI_Group w,e; MPI_Comm_group(MPI_COMM_WORLD,&w); int r=MPI_Group_incl(w,0,NULL,&e);
 printf("incl rc=%d empty?%d\n", r, e==MPI_GROUP_EMPTY);
 r=MPI_Group_free(&e); printf("free(GROUP_EMPTY) rc=%d\n", r);
 MPI_Group_free(&w); MPI_Finalize(); return 0; }
