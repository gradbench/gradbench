#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"
#include "codi_impl.hpp"


class Dir : public Function<kmeans::Input, kmeans::DirOutput>, CoDiReverseRunner2nd {
  using Real = typename CoDiReverseRunner2nd::Real;

  std::vector<Real> ad_centroids;
  Real err;

public:


  Dir(kmeans::Input& input) : Function(input),
    ad_centroids(_input.centroids.size()),
    err() {

    std::copy(_input.centroids.begin(), _input.centroids.end(),
              ad_centroids.begin());
  }

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    codiStartRecording();

    for (auto &v : ad_centroids) {
      codiAddInput(v);
    }

    kmeans::objective<Real>(_input.n, _input.k, _input.d,
                           _input.points.data(),
                           ad_centroids.data(),
                           &err);

    codiAddOutput(err);
    codiStopRecording();

    codiSetGradient(err, 1.0);
    codiEval();

    for (int i = 0; i < _input.k * _input.d; i++){
      output.dir[i] =
        codiGetGradient(ad_centroids[i], 1, 1)/
        codiGetGradient(ad_centroids[i], 2, 0);
    }

    codiCleanup();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"cost", function_main<kmeans::Cost>},
      {"dir", function_main<Dir>}
    });
}
