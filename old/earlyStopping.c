
#include <earlyStopping.h>
int patience = 10;

int Counter;

void  earlyStopping(){ 
    if (Counter==patience){
        return;
        }
        else if (Counter>patience){
            printf("Something is very wrong here \n");
        }
        else{
            printf("\n");
        }
} 