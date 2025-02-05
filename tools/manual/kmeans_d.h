#pragma once

#include "adbench/shared/kmeans.h"

void kmeans_objective_d(int n, int k, int d, double const *points,
                        double const *centroids,
                        double *dir);
