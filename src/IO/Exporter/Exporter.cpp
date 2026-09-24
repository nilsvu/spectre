// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/Exporter.h"
#include "IO/Exporter/Exporter.hpp"

#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "Utilities/GenerateInstantiations.hpp"

extern "C" void spectre_interpolate_to_points(
    double* out, const char** volume_files, size_t num_volume_files,
    const char* subfile_name, int observation_step,
    const char** tensor_components, size_t num_tensor_components,
    const double* target_points, size_t num_dimensions, size_t num_points,
    bool extrapolate_into_excisions, bool error_on_missing_points,
    int num_threads) {
  const auto impl = [&]<size_t Dim>() {
    tnsr::I<DataVector, Dim, Frame::Inertial> target_points_dv{};
    for (size_t d = 0; d < Dim; ++d) {
      target_points_dv.get(d).set_data_ref(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<double*>(target_points + d * num_points), num_points);
    }
    std::vector<std::vector<double>> result{};
    spectre::Exporter::interpolate_to_points(
        make_not_null(&result),
        std::vector<std::string>(volume_files, volume_files + num_volume_files),
        std::string(subfile_name),
        spectre::Exporter::ObservationStep(observation_step),
        std::vector<std::string>(tensor_components,
                                 tensor_components + num_tensor_components),
        target_points_dv, extrapolate_into_excisions, error_on_missing_points,
        num_threads < 0 ? std::nullopt : std::optional<size_t>(num_threads));
    for (size_t i = 0; i < result.size(); ++i) {
      std::copy(result[i].begin(), result[i].end(), out + i * num_points);
    }
  };
  if (num_dimensions == 1) {
    impl.template operator()<1>();
  } else if (num_dimensions == 2) {
    impl.template operator()<2>();
  } else if (num_dimensions == 3) {
    impl.template operator()<3>();
  } else {
    ERROR("Unsupported number of dimensions: " << num_dimensions);
  }
}

namespace spectre::Exporter {

template <size_t Dim>
std::vector<std::vector<double>> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const std::array<std::vector<double>, Dim>& target_points,
    const bool extrapolate_into_excisions, const bool error_on_missing_points,
    const std::optional<size_t> num_threads) {
  tnsr::I<DataVector, Dim, Frame::Inertial> target_points_dv{};
  for (size_t d = 0; d < Dim; ++d) {
    target_points_dv.get(d).set_data_ref(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(gsl::at(target_points, d).data()),
        gsl::at(target_points, d).size());
  }
  std::vector<std::vector<double>> result{};
  interpolate_to_points(make_not_null(&result), volume_files_or_glob,
                        subfile_name, observation, tensor_components,
                        target_points_dv, extrapolate_into_excisions,
                        error_on_missing_points, num_threads);
  return result;
}

// Generate instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::vector<std::vector<double>> interpolate_to_points<DIM(data)>( \
      const std::variant<std::vector<std::string>, std::string>&              \
          volume_files_or_glob,                                               \
      const std::string& subfile_name, const ObservationVariant& observation, \
      const std::vector<std::string>& tensor_components,                      \
      const std::array<std::vector<double>, DIM(data)>& target_points,        \
      bool extrapolate_into_excisions, bool error_on_missing_points,          \
      std::optional<size_t> num_threads);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace spectre::Exporter
