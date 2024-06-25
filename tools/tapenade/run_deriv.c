#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void square_b(double x, double *xb, double yb);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <value of x>\n", argv[0]);
        return 1;
    }
    double x = atof(argv[1]);

    double xb = 0.0;   // This will hold the derivative dy/dx
    double yb = 1.0;   // Set to 1 to compute the derivative

    struct timespec start, finish;

    clock_gettime( CLOCK_REALTIME, &start );
    square_b(x, &xb, yb);
    clock_gettime( CLOCK_REALTIME, &finish );

    double time_taken = (double) (finish.tv_nsec - start.tv_nsec);

    printf("%f %.0f", xb, time_taken);
}
