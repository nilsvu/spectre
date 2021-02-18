// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace elliptic::Actions {

template <typename System, typename BackgroundTag>
struct InitializeBackgroundFields {
 private:
  static constexpr size_t Dim = System::volume_dim;
  using background_fields_tag =
      ::Tags::Variables<typename System::background_fields>;

 public:
  using simple_tags = tmpl::list<background_fields_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& background = db::get<BackgroundTag>(box);
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& mesh = get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian = get<
        domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>>(
        box);
    auto background_fields =
        background.variables(inertial_coords, mesh, inv_jacobian,
                             typename background_fields_tag::tags_list{});
    ::Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(background_fields));
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
