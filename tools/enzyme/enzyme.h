#pragma once

// Shared declarations for the Enzyme C++ API.

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_autodiff(... ) noexcept;
void __enzyme_fwddiff(... ) noexcept;
