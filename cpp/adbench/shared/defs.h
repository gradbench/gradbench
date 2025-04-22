// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/defs.h

// Changes made: removed most things.

#pragma once

typedef struct {
  double gamma;
  int    m;
} Wishart;

typedef struct {
  int verts[3];
} Triangle;

#ifndef PI
#define PI 3.14159265359
#endif
