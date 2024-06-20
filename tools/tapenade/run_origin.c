#include <stdio.h>
#include <stdlib.h>

double square(double x);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <value of x>\n", argv[0]);
        return 1;  // Exit if no input is provided
    }
    double x = atof(argv[1]);

    // TIME THIS
    double y = square(x);
    printf("%.1f", y);
    return 0;
}
