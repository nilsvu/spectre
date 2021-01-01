// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "Elliptic/BoundaryConditions/Zero.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace elliptic::BoundaryConditions {

/// \cond
template <size_t Dim, typename FieldTags, typename Registrars>
struct AnalyticSolution;

namespace Registrars {
template <size_t Dim, typename FieldTags>
struct AnalyticSolution {
  template <typename Registrars>
  using f = BoundaryConditions::AnalyticSolution<Dim, FieldTags, Registrars>;

  using linearization_registrar = Zero<Dim, FieldTags>;
};
}  // namespace Registrars

template <size_t Dim, typename FieldTags,
          typename Registrars = tmpl::list<
              BoundaryConditions::Registrars::AnalyticSolution<Dim, FieldTags>>>
struct AnalyticSolution;
/// \endcond

template <size_t Dim, typename... FieldTags, typename Registrars>
class AnalyticSolution<Dim, tmpl::list<FieldTags...>, Registrars>
    : public BoundaryCondition<Dim, Registrars> {
 private:
  using Base = BoundaryCondition<Dim, Registrars>;

 public:
  using options =
      tmpl::list<elliptic::OptionTags::BoundaryConditionType<FieldTags>...>;
  static constexpr Options::String help =
      "Boundary conditions from an analytic solution";

  AnalyticSolution() = default;
  AnalyticSolution(const AnalyticSolution&) noexcept = default;
  AnalyticSolution& operator=(const AnalyticSolution&) noexcept = default;
  AnalyticSolution(AnalyticSolution&&) noexcept = default;
  AnalyticSolution& operator=(AnalyticSolution&&) noexcept = default;
  ~AnalyticSolution() noexcept = default;

  /// \cond
  explicit AnalyticSolution(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AnalyticSolution);  // NOLINT
  /// \endcond

  explicit AnalyticSolution(
      const typename elliptic::OptionTags::BoundaryConditionType<
          FieldTags>::type... boundary_condition_types) noexcept
      : boundary_condition_types_{boundary_condition_types...} {}

  std::unique_ptr<
      BoundaryConditionBase<Dim, typename Base::linearization_registrars>>
  linearization() const noexcept override {
    // The linearization is always zero since the boundary conditions are
    // independent of the dynamic fields
    return std::make_unique<Zero<Dim, tmpl::list<FieldTags...>,
                                 typename Base::linearization_registrars>>(
        get<elliptic::Tags::BoundaryConditionType<FieldTags>>(
            boundary_condition_types_)...);
  }

  const auto& boundary_condition_types() const noexcept {
    return boundary_condition_types_;
  }

  // Argument tags are taken from exterior faces, i.e. the face normal points
  // into the domain.
  using argument_tags =
      tmpl::list<::Tags::AnalyticSolutionsBase, domain::Tags::Mesh<Dim>,
                 domain::Tags::Direction<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>>;
  using volume_tags =
      tmpl::list<::Tags::AnalyticSolutionsBase, domain::Tags::Mesh<Dim>>;

  template <typename OptionalAnalyticSolutions>
  void operator()(
      const OptionalAnalyticSolutions& optional_analytic_solutions,
      const Mesh<Dim>& volume_mesh, const Direction<Dim>& direction,
      const tnsr::i<DataVector, Dim>& face_normal,
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... n_dot_fluxes) const
      noexcept {
    const auto& analytic_solutions = [&optional_analytic_solutions]() {
      if constexpr (tt::is_a_v<std::optional, OptionalAnalyticSolutions>) {
        if (not optional_analytic_solutions.has_value()) {
          ERROR(
              "The background does not provide an analytic solution. Select an "
              "analytic solution as background if you want to use it to impose "
              "boundary conditions.");
        }
        return *optional_analytic_solutions;
      } else {
        return optional_analytic_solutions;
      }
    }();
    const size_t slice_index =
        index_to_slice_at(volume_mesh.extents(), direction);
    const auto helper = [this, &analytic_solutions, &volume_mesh, &direction,
                         &slice_index,
                         &face_normal](auto field_tag_v, const auto field,
                                       const auto n_dot_flux) noexcept {
      using field_tag = decltype(field_tag_v);
      switch (get<elliptic::Tags::BoundaryConditionType<field_tag>>(
          boundary_condition_types())) {
        case elliptic::BoundaryConditionType::Dirichlet:
          data_on_slice(
              field, get<::Tags::Analytic<field_tag>>(analytic_solutions),
              volume_mesh.extents(), direction.dimension(), slice_index);
          break;
        case elliptic::BoundaryConditionType::Neumann:
          normal_dot_flux(
              n_dot_flux, face_normal,
              data_on_slice(
                  get<::Tags::Analytic<::Tags::Flux<
                      field_tag, tmpl::size_t<Dim>, Frame::Inertial>>>(
                      analytic_solutions),
                  volume_mesh.extents(), direction.dimension(), slice_index));
          break;
        default:
          ERROR("Unsupported boundary condition type: "
                << get<elliptic::Tags::BoundaryConditionType<field_tag>>(
                       boundary_condition_types()));
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(FieldTags{}, fields, n_dot_fluxes));
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    p | boundary_condition_types_;
  }

 private:
  tuples::TaggedTuple<elliptic::Tags::BoundaryConditionType<FieldTags>...>
      boundary_condition_types_{};
};

/// \cond
template <size_t Dim, typename... FieldTags, typename Registrars>
PUP::able::PUP_ID
    AnalyticSolution<Dim, tmpl::list<FieldTags...>, Registrars>::my_PUP_ID =
        0;  // NOLINT
/// \endcond

}  // namespace elliptic::BoundaryConditions
