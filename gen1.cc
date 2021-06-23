// Generate symmetric matrix with n on diagonal, n-1 on sub-diagonal...
// 4 3 2 1
// 3 4 3 2
// 2 3 4 3
// 1 2 3 4

#include <iostream>

#include <string.h>

#include <stdio.h>

#include <stdlib.h>

#include <math.h>


int main(int argc, char * argv[]) {
    int lines = atoi(argv[1]);
    printf("%d,\n", lines);
    for (int i = 0; i < lines; i++) {
        for (int j = 0; j < lines; j++) {
            printf("%d,", lines - abs(j - i));
        }
        printf("\n");
    }
    printf("\n");
}
