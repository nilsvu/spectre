// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

namespace InitializeAnalyticSolution_detail {
template <size_t Dim, typename AnalyticSolutionFields,
          typename AnalyticSolution>
void initialize_analytic_solution(
    const gsl::not_null<
        Variables<db::wrap_tags_in<::Tags::Analytic, AnalyticSolutionFields>>*>
        analytic_fields,
    const gsl::not_null<std::unordered_map<
        Direction<Dim>,
        Variables<db::wrap_tags_in<::Tags::Analytic, AnalyticSolutionFields>>>*>
        analytic_fields_on_external_faces,
    const AnalyticSolution& analytic_solution, const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim>& inertial_coords,
    const std::unordered_map<Direction<Dim>, tnsr::I<DataVector, Dim>>&
        inertial_coords_on_external_faces) noexcept {
  // Analytic solution in the volume
  *analytic_fields = variables_from_tagged_tuple(
      analytic_solution.variables(inertial_coords, AnalyticSolutionFields{}));
  // Analytic solution on the external boundary faces, for boundary conditions
  for (const auto& [direction, inertial_coords_on_face] :
       inertial_coords_on_external_faces) {
    auto& analytic_fields_on_face =
        (*analytic_fields_on_external_faces)[direction];
    if (mesh.quadrature(direction.dimension()) ==
        Spectral::Quadrature::GaussLobatto) {
      // Slice the boundary data from the volume so we don't have to re-evaluate
      // the analytic solution
      data_on_slice(make_not_null(&analytic_fields_on_face), *analytic_fields,
                    mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
    } else {
      // Evaluate the analytic solution on the boundary, because the volume data
      // doesn't have points on the boundary
      (*analytic_fields_on_external_faces)[direction] =
          variables_from_tagged_tuple(analytic_solution.variables(
              inertial_coords_on_face, AnalyticSolutionFields{}));
    }
  }
}
}  // namespace InitializeAnalyticSolution_detail

// @{
/*!
 * \brief Place the analytic solution of the system fields in the DataBox.
 *
 * Use `InitializeAnalyticSolution` if it is clear at compile-time that an
 * analytic solution is available, and `InitializeOptionalAnalyticSolution` if
 * that is a runtime decision, e.g. based on a choice in the input file. The
 * `::Tags::AnalyticSolutionsBase` and `::Tags::AnalyticSolutionsOnBoundaryBase`
 * tags can be retrieved from the DataBox in either case, but they will hold a
 * `std::optional` when `InitializeOptionalAnalyticSolution` is used. In that
 * case, the analytic solution is only evaluated and stored in the DataBox if
 * the `BackgroundTag` holds a type that inherits from the
 * `AnalyticSolutionType`.
 *
 * Uses:
 * - DataBox:
 *   - `AnalyticSolutionTag` or `BackgroundTag`
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
 *     domain::Tags::Coordinates<Dim, Frame::Inertial>>`
 *
 * DataBox:
 * - Adds:
 *   - `::Tags::AnalyticSolutionsBase`
 *   - `::Tags::AnalyticSolutionsOnBoundaryBase`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <size_t Dim, typename AnalyticSolutionTag,
          typename AnalyticSolutionFields>
struct InitializeAnalyticSolution {
 private:
  using analytic_fields_tag = ::Tags::AnalyticSolutions<AnalyticSolutionFields>;
  using analytic_fields_on_boundary_tag =
      ::Tags::AnalyticSolutionsOnBoundary<Dim, AnalyticSolutionFields>;

 public:
  using simple_tags =
      tmpl::list<analytic_fields_tag, analytic_fields_on_boundary_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // The `AnalyticSolutionTag` surely holds an analytic solution, so we can
    // invoke it directly
    db::mutate_apply<
        simple_tags,
        tmpl::list<AnalyticSolutionTag, domain::Tags::Mesh<Dim>,
                   domain::Tags::Coordinates<Dim, Frame::Inertial>,
                   domain::Tags::Interface<
                       domain::Tags::BoundaryDirectionsInterior<Dim>,
                       domain::Tags::Coordinates<Dim, Frame::Inertial>>>>(
        InitializeAnalyticSolution_detail::initialize_analytic_solution<
            Dim, AnalyticSolutionFields,
            std::decay_t<decltype(db::get<AnalyticSolutionTag>(box))>>,
        make_not_null(&box));
    return {std::move(box)};
  }
};

template <size_t Dim, typename BackgroundTag, typename AnalyticSolutionFields,
          typename AnalyticSolutionType>
struct InitializeOptionalAnalyticSolution {
 private:
  using analytic_fields_tag =
      ::Tags::AnalyticSolutionsOptional<AnalyticSolutionFields>;
  using analytic_fields_on_boundary_tag =
      ::Tags::AnalyticSolutionsOnBoundaryOptional<Dim, AnalyticSolutionFields>;

 public:
  using simple_tags =
      tmpl::list<analytic_fields_tag, analytic_fields_on_boundary_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // The `BackgroundTag` may or may not be an analytic solution. We need to
    // check.
    const auto analytic_solution =
        dynamic_cast<const AnalyticSolutionType*>(&db::get<BackgroundTag>(box));
    if (analytic_solution != nullptr) {
      db::mutate_apply<
          simple_tags,
          tmpl::list<domain::Tags::Mesh<Dim>,
                     domain::Tags::Coordinates<Dim, Frame::Inertial>,
                     domain::Tags::Interface<
                         domain::Tags::BoundaryDirectionsInterior<Dim>,
                         domain::Tags::Coordinates<Dim, Frame::Inertial>>>>(
          [&analytic_solution](const auto analytic_fields,
                               const auto analytic_fields_on_boundary,
                               const auto&... args) noexcept {
            *analytic_fields = typename analytic_fields_tag::type::value_type{};
            *analytic_fields_on_boundary =
                typename analytic_fields_on_boundary_tag::type::value_type{};
            InitializeAnalyticSolution_detail::initialize_analytic_solution<
                Dim, AnalyticSolutionFields>(
                make_not_null(&analytic_fields->value()),
                make_not_null(&analytic_fields_on_boundary->value()),
                *analytic_solution, args...);
          },
          make_not_null(&box));
    }
    return {std::move(box)};
  }
};
// @}
}  // namespace elliptic::Actions
