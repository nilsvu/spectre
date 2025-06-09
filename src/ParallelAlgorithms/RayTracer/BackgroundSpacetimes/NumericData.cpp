// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/NumericData.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "IO/Exporter/SpacetimeInterpolator.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ray_tracing {

NumericData::NumericData(std::string file_glob, std::string subfile_name,
                         std::optional<int> observation_step,
                        ::Verbosity verbosity)
    : file_glob_(std::move(file_glob)),
      subfile_name_(std::move(subfile_name)),
      observation_step_(observation_step),
      verbosity_(verbosity) {
  if (not observation_step_.has_value()) {
    // Construct the spacetime interpolator but don't load any data yet
    interpolator_ = spectre::Exporter::SpacetimeInterpolator<Dim, Frame>{
        file_glob_, subfile_name_,
        spectre::Exporter::get_tensor_components<tags>()};
  }
}

NumericData::NumericData(const NumericData& rhs)
    : NumericData(rhs.file_glob_, rhs.subfile_name_, rhs.observation_step_,
    rhs.verbosity_) {}

NumericData& NumericData::operator=(const NumericData& rhs) {
  if (this != &rhs) {
    *this = NumericData(rhs);
  }
  return *this;
}

void NumericData::initialize(std::array<double, 2> new_time_bounds) {
  if (verbosity_ >= ::Verbosity::Verbose) {
  Parallel::printf("Loading numeric data...\n");
  }
  if (observation_step_.has_value()) {
    interpolator_ = spectre::Exporter::PointwiseInterpolator<Dim, Frame>{
        file_glob_, subfile_name_,
        spectre::Exporter::ObservationStep{observation_step_.value()},
        spectre::Exporter::get_tensor_components<tags>()};
  } else {
    std::get<spectre::Exporter::SpacetimeInterpolator<Dim, Frame>>(
        interpolator_)
        .load_time_bounds(new_time_bounds);
  }
  if (verbosity_ >= ::Verbosity::Verbose) {
    Parallel::printf("Numeric data loaded.\n");
  }
}

std::array<double, 2> NumericData::time_bounds() const {
  if (std::holds_alternative<
          spectre::Exporter::SpacetimeInterpolator<Dim, Frame>>(
          interpolator_)) {
    return std::get<spectre::Exporter::SpacetimeInterpolator<Dim, Frame>>(
               interpolator_)
        .time_bounds();
  } else {
    return {{-std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()}};
  }
}

tuples::tagged_tuple_from_typelist<typename NumericData::tags>
NumericData::variables(const tnsr::I<DataType, Dim, Frame>& x, const double t,
                       const std::optional<gsl::not_null<std::vector<size_t>*>>
                           block_order) const {
  std::vector<double> result{};
  std::visit(
      Overloader{[&x, &result, &block_order](
                     const spectre::Exporter::PointwiseInterpolator<Dim, Frame>&
                         interp) {
                   return interp.interpolate_to_point(make_not_null(&result), x,
                                                      block_order);
                 },
                 [&x, &t, &result, &block_order](
                     const spectre::Exporter::SpacetimeInterpolator<Dim, Frame>&
                         interp) {
                   return interp.interpolate_to_point(make_not_null(&result), x,
                                                      t, block_order);
                 }},
      interpolator_);
  return spectre::Exporter::make_tagged_tuple<tags>(result);
}

void NumericData::pup(PUP::er& p) {
  BackgroundSpacetime::pup(p);
  p | file_glob_;
  p | subfile_name_;
  p | observation_step_;
  // Don't copy interpolator, it must be reinitialized
}

PUP::able::PUP_ID NumericData::my_PUP_ID = 0;

}  // namespace ray_tracing
