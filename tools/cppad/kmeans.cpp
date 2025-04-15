#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
  CppAD::ADFun<double> *_tape;
  std::vector<double> _H, _J;
  CppAD::sparse_hessian_work _work;
  std::vector<bool> _p;
  std::vector<size_t> _row, _col;
  std::vector<ADdouble> _acentroids;
public:
  Dir(kmeans::Input& input)
    : Function(input),
      _H(input.k*input.d),
      _J(input.k*input.d),
      _p((input.k*input.d)*(input.k*input.d)),
      _row(input.k*input.d),
      _col(input.k*input.d),
      _acentroids(_input.centroids.size())
  {
    std::copy(_input.centroids.begin(),
              _input.centroids.end(),
              _acentroids.data());


    size_t input_size = _input.k * _input.d;

    // Compute sparsity pattern - it is unclear to me whether this is
    // used when we also pass 'col' and 'row' arguments in the call to
    // SparseHessian.
    for (size_t i = 0; i < input_size; i++) {
      for (size_t j = 0; j < input_size; j++) {
        _p[i*input_size+j] = i==j;
      }
    }

    // Provide row and column information. We only wish to compute the
    // diagonal.
    for (size_t i = 0; i < input_size; i++) {
      _row[i] = _col[i] = i;
    }
  }

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    CppAD::Independent(_acentroids);

    std::vector<ADdouble> err(1);

    kmeans::objective<ADdouble>(_input.n, _input.k, _input.d,
                                _input.points.data(), _acentroids.data(),
                                &err[0]);

    CppAD::ADFun<double> f(_acentroids, err);
    f.optimize("no_compare_op no_conditional_skip no_print_for_op");

    _J = f.Jacobian(_input.centroids);

    std::vector<double> w(1);
    w[0] = 1;

    size_t nsweep =
      f.SparseHessian(_input.centroids, w, _p, _row, _col, _H, _work);
    assert(nsweep == 1); // Just a precaution.

    for (int i = 0; i < _input.k * _input.d; i++) {
      output.dir[i] = _J[i] / _H[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"cost", function_main<kmeans::Cost>},
      {"dir", function_main<Dir>}
    });
}
