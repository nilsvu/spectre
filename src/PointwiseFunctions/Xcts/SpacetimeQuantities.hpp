// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {
/// \cond
struct SpacetimeQuantitiesComputer;
/// \endcond

namespace detail {
template <typename DataType>
struct ExtrinsicCurvatureSquare : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace detail

/// General-relativistic 3+1 quantities computed from XCTS variables.
using SpacetimeQuantities = CachedTempBuffer<
    SpacetimeQuantitiesComputer,
    gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
    ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
    ::Tags::deriv<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
        tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::SpatialRicci<3, Frame::Inertial, DataVector>,
    gr::Tags::Lapse<DataVector>,
    gr::Tags::Shift<3, Frame::Inertial, DataVector>,
    ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>,
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
    detail::ExtrinsicCurvatureSquare<DataVector>,
    ::Tags::deriv<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::HamiltonianConstraint<DataVector>,
    gr::Tags::MomentumConstraint<3, Frame::Inertial, DataVector>>;

/// `CachedTempBuffer` computer class for 3+1 quantities from XCTS variables.
/// See `Xcts::SpacetimeQuantities`.
struct SpacetimeQuantitiesComputer {
  using Cache = SpacetimeQuantities;

  void operator()(gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SpatialMetric<3, Frame::Inertial,
                                          DataVector> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::II<DataVector, 3>*> inv_spatial_metric,
      gsl::not_null<Cache*> cache,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::ijj<DataVector, 3>*> deriv_spatial_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::Ijj<DataVector, 3>*> christoffel_second_kind,
      gsl::not_null<Cache*> cache,
      gr::Tags::SpatialChristoffelSecondKind<
          3, Frame::Inertial, DataVector> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::iJkk<DataVector, 3>*> deriv_christoffel_second_kind,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                           DataVector>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::ii<DataVector, 3>*> ricci_tensor,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SpatialRicci<3, Frame::Inertial,
                                         DataVector> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataVector>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataVector> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataVector, 3>*> shift, gsl::not_null<Cache*> cache,
      gr::Tags::Shift<3, Frame::Inertial, DataVector> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::iJ<DataVector, 3>*> deriv_shift,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::ii<DataVector, 3>*> dt_spatial_metric,
                  gsl::not_null<Cache*> cache,
                  ::Tags::dt<gr::Tags::SpatialMetric<
                      3, Frame::Inertial, DataVector>> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::ii<DataVector, 3>*> extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataVector>*> extrinsic_curvature_square,
      gsl::not_null<Cache*> cache,
      detail::ExtrinsicCurvatureSquare<DataVector> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::ijj<DataVector, 3>*> deriv_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<
          gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
          tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      gsl::not_null<Cache*> cache,
      gr::Tags::HamiltonianConstraint<DataVector> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataVector, 3>*> momentum_constraint,
      gsl::not_null<Cache*> cache,
      gr::Tags::MomentumConstraint<3, Frame::Inertial, DataVector> /*meta*/)
      const noexcept;

  const Scalar<DataVector>& conformal_factor;
  const Scalar<DataVector>& lapse_times_conformal_factor;
  const tnsr::I<DataVector, 3>& shift_excess;
  const tnsr::ii<DataVector, 3>& conformal_metric;
  const tnsr::II<DataVector, 3>& inv_conformal_metric;
  const tnsr::I<DataVector, 3>& shift_background;
  const Mesh<3>& mesh;
  const InverseJacobian<DataVector, 3, Frame::Logical, Frame::Inertial>&
      inv_jacobian;
};

namespace OptionTags {
struct Constraints {
  static constexpr Options::String help = "Options for computing constraints";
};
template <size_t Dim>
struct OversampleMesh : db::SimpleTag {
  using type = Options::Auto<::Mesh<Dim>, Options::AutoLabel::None>;
  using group = Constraints;
  static constexpr Options::String help = "Compute constraints on this mesh.";
};
}  // namespace OptionTags

namespace Tags {

template <size_t Dim>
struct ConstraintsOversampleMesh : db::SimpleTag {
  using type = std::optional<::Mesh<Dim>>;
  using option_tags = tmpl::list<OptionTags::OversampleMesh<Dim>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) noexcept { return value; }
};

/// Compute tag for the 3+1 quantities `Tags` from XCTS variables. The `Tags`
/// can be any subset of the tags supported by `Xcts::SpacetimeQuantities`.
template <typename BackgroundTag, typename Tags>
struct SpacetimeQuantitiesCompute : ::Tags::Variables<Tags>, db::ComputeTag {
 private:
  using vars_tags = tmpl::list<ConformalFactor<DataVector>,
                               LapseTimesConformalFactor<DataVector>,
                               ShiftExcess<DataVector, 3, Frame::Inertial>>;
  using background_tags =
      tmpl::list<ConformalMetric<DataVector, 3, Frame::Inertial>,
                 InverseConformalMetric<DataVector, 3, Frame::Inertial>,
                 ShiftBackground<DataVector, 3, Frame::Inertial>>;

 public:
  using base = ::Tags::Variables<Tags>;
  using argument_tags =
      tmpl::list<::Tags::Variables<vars_tags>, domain::Tags::Mesh<3>,
                 domain::Tags::ElementMap<3>, ConstraintsOversampleMesh<3>,
                 BackgroundTag>;
  template <typename Background>
  static void function(const gsl::not_null<typename base::type*> result,
                       const ::Variables<vars_tags>& original_vars,
                       const Mesh<3>& original_mesh,
                       const ElementMap<3, Frame::Inertial>& element_map,
                       const std::optional<Mesh<3>>& oversample_mesh,
                       const Background& background) noexcept {
    const Mesh<3>& mesh = oversample_mesh.value_or(original_mesh);
    const size_t num_points = mesh.number_of_grid_points();
    if (result->number_of_grid_points() != num_points) {
      result->initialize(num_points);
    }
    // Interpolate solved variables to oversampled mesh
    const intrp::RegularGrid<3> interpolant{original_mesh, mesh};
    const auto vars = interpolant.interpolate(original_vars);
    // Evaluate background quantities on oversampled mesh
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = element_map(logical_coords);
    const auto inv_jacobian = element_map.inv_jacobian(logical_coords);
    const auto background_quantities =
        background.variables(inertial_coords, background_tags{});
    // Compute spacetime quantities
    SpacetimeQuantities spacetime_quantities{
        num_points,
        {get<ConformalFactor<DataVector>>(vars),
         get<LapseTimesConformalFactor<DataVector>>(vars),
         get<ShiftExcess<DataVector, 3, Frame::Inertial>>(vars),
         get<ConformalMetric<DataVector, 3, Frame::Inertial>>(
             background_quantities),
         get<InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
             background_quantities),
         get<ShiftBackground<DataVector, 3, Frame::Inertial>>(
             background_quantities),
         mesh, inv_jacobian}};
    tmpl::for_each<Tags>(
        [&spacetime_quantities, &result](const auto tag_v) noexcept {
          using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
          get<tag>(*result) = spacetime_quantities.get_var(tag{});
        });
  }
};
}  // namespace Tags

}  // namespace Xcts
