#include <vector>

#include "ad.hpp"
#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"

using tangent_t  = ad::tangent_t<double>;
using tangent2_t = ad::tangent_t<tangent_t>;

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {

public:
  Dir(kmeans::Input& input) : Function(input) {}

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);
    std::vector<tangent2_t> centroids_active(_input.centroids.size());
    for (size_t i = 0; i < centroids_active.size(); i++) {
      ad::passive_value(centroids_active[i])              = _input.centroids[i];
      ad::derivative(ad::derivative(centroids_active[i])) = 0;
      ad::value(ad::derivative(centroids_active[i]))      = 0;
      ad::derivative(ad::value(centroids_active[i]))      = 0;
    }

    tangent2_t active_err;
    for (size_t i = 0; i < centroids_active.size(); i++) {
      // set x^(1) and x^(2)
      ad::value(ad::derivative(centroids_active[i])) = 1;
      ad::derivative(ad::value(centroids_active[i])) = 1;

      kmeans::objective(_input.n, _input.k, _input.d, _input.points.data(),
                        centroids_active.data(), &active_err);
      // compute J_i * H^-1_{ii} and write to out.dir[i]
      output.dir[i] = ad::derivative(ad::value(active_err)) /
                      ad::derivative(ad::derivative(active_err));
      // reset x^(1) and x^(2)
      ad::value(ad::derivative(centroids_active[i])) = 0;
      ad::derivative(ad::value(centroids_active[i])) = 0;
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(
      argc, argv,
      {{"cost", function_main<kmeans::Cost>}, {"dir", function_main<Dir>}});
}
