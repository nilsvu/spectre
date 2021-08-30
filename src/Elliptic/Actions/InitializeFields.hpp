// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Amr/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::Actions {

/*!
 * \brief Initialize the dynamic fields of the elliptic system, i.e. those we
 * solve for.
 *
 * Uses:
 * - System:
 *   - `primal_fields`
 * - DataBox:
 *   - `InitialGuessTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `primal_fields`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename System, typename InitialGuessTag,
          typename ArraySectionIdTag = void>
struct InitializeFields {
 private:
  using fields_tag = ::Tags::Variables<typename System::primal_fields>;

 public:
  using simple_tags = tmpl::list<fields_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Skip initialization on elements that are not part of the section (e.g. on
    // coarser multigrid levels)
    if constexpr (not std::is_same_v<ArraySectionIdTag, void>) {
      if (not db::get<Parallel::Tags::Section<ParallelComponent,
                                              ArraySectionIdTag>>(box)
                  .has_value()) {
        db::mutate<fields_tag>(
            make_not_null(&box),
            [](const auto fields, const Mesh<Dim>& mesh) noexcept {
              fields->initialize(mesh.number_of_grid_points());
            },
            db::get<domain::Tags::Mesh<Dim>>(box));
        return {std::move(box)};
      }
    }
    // At higher AMR levels, get initial guess by interpolation from previous
    // AMR level. Currently only p-AMR is supported.
    if constexpr (db::tag_is_retrievable_v<elliptic::amr::Tags::Level,
                                           db::DataBox<DbTagsList>>) {
      const std::optional<Mesh<Dim>>& amr_parent_mesh =
          db::get<elliptic::amr::Tags::ParentMesh<Dim>>(box);
      if (amr_parent_mesh.has_value()) {
        db::mutate<fields_tag>(
            make_not_null(&box),
            [&amr_parent_mesh](const auto fields,
                               const Mesh<Dim>& mesh) noexcept {
              const intrp::RegularGrid<Dim> interpolant{amr_parent_mesh.value(),
                                                        mesh};
              const auto initial_fields = interpolant.interpolate(*fields);
              *fields = initial_fields;
            },
            db::get<domain::Tags::Mesh<Dim>>(box));
        return {std::move(box)};
      }
    }
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& initial_guess = db::get<InitialGuessTag>(box);
    auto initial_fields = variables_from_tagged_tuple(initial_guess.variables(
        inertial_coords, typename fields_tag::tags_list{}));
    ::Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(initial_fields));
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
