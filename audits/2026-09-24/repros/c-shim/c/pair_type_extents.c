#include <mpi.h>
#include <stdio.h>
int main(int c,char**v){MPI_Init(&c,&v);
 MPI_Datatype t[6]={MPI_FLOAT_INT,MPI_DOUBLE_INT,MPI_LONG_INT,MPI_2INT,MPI_SHORT_INT,MPI_LONG_DOUBLE_INT};
 const char*n[6]={"FLOAT_INT","DOUBLE_INT","LONG_INT","2INT","SHORT_INT","LONG_DOUBLE_INT"};
 for(int i=0;i<6;i++){MPI_Aint lb,ex;int sz;MPI_Type_get_extent(t[i],&lb,&ex);MPI_Type_size(t[i],&sz);printf("%s lb=%ld extent=%ld size=%d\n",n[i],(long)lb,(long)ex,sz);}
 MPI_Finalize();}
