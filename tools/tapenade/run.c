#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void square(double x, double *y);
void square_b(double x, double *xb, double *y, double *yb);

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <value of x>\n", argv[0]);
        return 1;  // Exit if no input is provided
    }
    double x = atof(argv[1]);
    char* func = argv[2];

    double y;          // Function result
    double xb = 0.0;   // This will hold the derivative dy/dx
    double yb = 1.0;   // Set to 1 to compute the derivative

    if (strcmp(func,"square") == 0){
        square(x, &y);
        printf("%.1f", y);
    }
    else{
        square(x, &y);
        square_b(x, &xb, &y, &yb);
        printf("%.1f", xb);
    }
    return 0;
}
