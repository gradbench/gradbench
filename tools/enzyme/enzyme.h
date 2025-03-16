#pragma once

// Shared declarations for the Enzyme C++ API.

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_autodiff(... ) noexcept;
void __enzyme_fwddiff(... ) noexcept;

template<typename return_type, typename ... T>
return_type __enzyme_autodiff_template(void*, T ... );

template<typename return_type, typename ... T>
return_type __enzyme_fwddiff_template(void*, T ... );
