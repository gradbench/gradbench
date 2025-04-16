#include "gradbench/evals/ba.hpp"
#include "gradbench/main.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Jacobian : public Function<ba::Input, ba::JacOutput> {
public:
  Jacobian(ba::Input& input) : Function(input) {}

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    const int cols = BA_NCAMPARAMS + 3 + 1;

    std::vector<ADdouble> X(cols);
    std::vector<double>   input_flat(cols);
    std::vector<ADdouble> Y(2);

    for (int i = 0; i < _input.p; i++) {
      const int camIdx = _input.obs[i * 2 + 0];
      const int ptIdx  = _input.obs[i * 2 + 1];

      std::copy(&_input.cams[camIdx * BA_NCAMPARAMS],
                &_input.cams[(camIdx + 1) * BA_NCAMPARAMS], &input_flat[0]);
      std::copy(&_input.X[ptIdx * 3], &_input.X[(ptIdx + 1) * 3],
                &input_flat[BA_NCAMPARAMS]);
      input_flat[BA_NCAMPARAMS + 3] = _input.w[i];

      std::copy(input_flat.begin(), input_flat.end(), X.begin());

      CppAD::Independent(X);

      ba::computeReprojError<ADdouble>(X.data(), X.data() + BA_NCAMPARAMS,
                                       X.data() + BA_NCAMPARAMS + 3,
                                       &_input.feats[i * 2], Y.data());

      CppAD::ADFun<double> f(X, Y);

      std::vector<double> gradient(cols * 2);
      std::vector<double> gradient_tr = f.Jacobian(input_flat);

      for (int i = 0; i < cols; i++) {
        gradient[i * 2]     = gradient_tr[i];
        gradient[i * 2 + 1] = gradient_tr[i + cols];
      }

      output.insert_reproj_err_block(i, camIdx, ptIdx, gradient.data());
    }

    for (int i = 0; i < _input.p; i++) {
      std::vector<double>   w = {_input.w[0]};
      std::vector<ADdouble> X = {w[0]};
      CppAD::Independent(X);
      std::vector<ADdouble> Y(1);

      ba::computeZachWeightError<ADdouble>(&X[0], &Y[0]);

      CppAD::ADFun<double> f(X, Y);

      output.insert_w_err_block(i, f.ForOne(w, 0)[0]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ba::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
