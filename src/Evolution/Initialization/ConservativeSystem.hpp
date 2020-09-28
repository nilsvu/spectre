// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // Needs to be included somewhere and here seems most natural.
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame

namespace domain {
namespace Tags {
template <size_t VolumeDim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
// IWYU pragma: no_forward_declare db::DataBox

namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Allocate variables needed for evolution of conservative systems
///
/// Uses:
/// - DataBox:
///   * `Tags::Mesh<Dim>`
///
/// DataBox changes:
/// - Adds:
///   * System::variables_tag
///   * db::add_tag_prefix<Tags::Flux, System::variables_tag>
///   * db::add_tag_prefix<Tags::Source, System::variables_tag>
///
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename Metavariables>
struct ConservativeSystem {
 private:
  using system = typename Metavariables::system;
  static_assert(system::is_in_flux_conservative_form,
                "System is not in flux conservative form");

  static constexpr size_t dim = system::volume_dim;

  using variables_tag = typename system::variables_tag;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, variables_tag,
                                         tmpl::size_t<dim>, Frame::Inertial>;
  using sources_tag = db::add_tag_prefix<::Tags::Source, variables_tag>;

  template <typename System, typename enable = std::true_type>
  struct simple_tags_impl {
    using type = tmpl::list<variables_tag, fluxes_tag, sources_tag>;
  };

  template <typename System>
  struct simple_tags_impl<
      System, std::bool_constant<System::has_primitive_and_conservative_vars>> {
    using type = tmpl::list<variables_tag, fluxes_tag, sources_tag,
                            typename system::primitive_variables_tag,
                            typename Metavariables::equation_of_state_tag>;
  };

 public:
  using simple_tags = typename simple_tags_impl<system>::type;

  using compute_tags = db::AddComputeTags<>;

  template <typename DbTagsList, typename... InboxTags,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t num_grid_points =
        db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points();
    Parallel::printf("grrr %zu\n", num_grid_points);
    typename variables_tag::type vars(num_grid_points);
    typename fluxes_tag::type fluxes(num_grid_points);
    typename sources_tag::type sources(num_grid_points);

    if constexpr (system::has_primitive_and_conservative_vars) {
      using PrimitiveVars = typename system::primitive_variables_tag::type;

      PrimitiveVars primitive_vars{
          db::get<domain::Tags::Mesh<dim>>(box).number_of_grid_points()};
      auto equation_of_state =
          db::get<::Tags::AnalyticSolutionOrData>(box).equation_of_state();
      db::mutate_assign(make_not_null(&box), simple_tags{}, std::move(vars),
                        std::move(fluxes), std::move(sources),
                        std::move(primitive_vars),
                        std::move(equation_of_state));
    } else {
      db::mutate_assign(make_not_null(&box), simple_tags{}, std::move(vars),
                         std::move(fluxes), std::move(sources));
    }
    return std::make_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Initialization
