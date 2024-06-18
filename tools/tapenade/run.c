#include <stdio.h>

void square(double x, double *y);
void square_b(double x, double *xb, double *y, double *yb);

int main() {
    double x = 3.0;    // Input from JSON
    
    double y;          // Function result
    double xb = 0.0;   // This will hold the derivative dy/dx
    double yb = 1.0;   // Set to 1 to compute the derivative

    square(x, &y);
    square_b(x, &xb, &y, &yb);

    // Print the results
    printf("y = %f\n", y);
    printf("dy/dx = %f\n", xb);

    return 0;
}
