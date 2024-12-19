#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double square(double x);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <value of x>\n", argv[0]);
        return 1;
    }
    double x = atof(argv[1]);

    struct timespec start, finish;
    clock_gettime( CLOCK_REALTIME, &start );
    double y = square(x);
    clock_gettime( CLOCK_REALTIME, &finish );

    double time_taken = (double) (finish.tv_nsec - start.tv_nsec);
    printf("%f\n{\"name\": \"evaluate\", \"nanoseconds\": %.0f}\n", y, time_taken);
    return 0;
}
