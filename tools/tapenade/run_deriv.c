#include <stdio.h>
#include <stdlib.h>

void square_b(double x, double *xb, double yb);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <value of x>\n", argv[0]);
        return 1;  // Exit if no input is provided
    }
    double x = atof(argv[1]);

    double xb = 0.0;   // This will hold the derivative dy/dx
    double yb = 1.0;   // Set to 1 to compute the derivative

    // TIME THIS
    square_b(x, &xb, yb);
    printf("%.1f", xb);
}
