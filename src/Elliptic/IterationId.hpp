// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <iosfwd>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elliptic {

/*!
 * \brief The factor to divide the `Elliptic::IterationId::value()` by to
 * recover the step number of the `ComponentTag`
 *
 * \details Since we must encode the `Elliptic::IterationId` in a single
 * number to identify observations, we embed each component ...
 */
template <typename ComponentTag>
constexpr size_t iteration_id_value_factor = 0.;
template <>
constexpr size_t iteration_id_value_factor<LinearSolver::Tags::IterationId> = 1;
template <>
constexpr size_t iteration_id_value_factor<NonlinearSolver::Tags::IterationId> =
    1e6;

/*!
 * \brief Identifies a step in an elliptic solve
 *
 */
template <typename... ComponentTags>
struct IterationId : tuples::TaggedTuple<ComponentTags...> {
  using tuples::TaggedTuple<ComponentTags...>::TaggedTuple;

  /*!
   * \brief Encodes all component `value`s to identify an observation by a
   * single number
   *
   * \details For each component we allocate a fixed range of integers specified
   *
   * For now assume the max number of steps for each component, then we can
   * reconstruct from the double. Eventually, use one size_t to identify the
   * total step, and construct it from each component's number of sub-steps
   * (which must then be kept track of).
   */
  double value() const noexcept {
    size_t v = 0;
    tmpl::for_each<tmpl::list<ComponentTags...>>([&v, this ](
        auto component_tag) noexcept {
      using ComponentTag = tmpl::type_from<decltype(component_tag)>;
      v += iteration_id_value_factor<ComponentTag> * get<ComponentTag>(*this);
    });
    return v;
  }

  IterationId<ComponentTags...> next() const noexcept {
    IterationId<ComponentTags...> result{*this};
    get<LinearSolver::Tags::IterationId>(result)++;
    return result;
  }
};

template <typename... ComponentTags>
size_t hash_value(const IterationId<ComponentTags...>& id) noexcept {
  size_t h = 0;
  tmpl::for_each<tmpl::list<ComponentTags...>>(
      [&h, &id ](auto component_tag) noexcept {
        using ComponentTag = tmpl::type_from<decltype(component_tag)>;
        boost::hash_combine(h, get<ComponentTag>(id));
      });
  return h;
}

}  // namespace Elliptic

namespace std {
template <typename... ComponentTags>
struct hash<Elliptic::IterationId<ComponentTags...>> {
  size_t operator()(const Elliptic::IterationId<ComponentTags...>& id) const
      noexcept {
    return boost::hash<Elliptic::IterationId<ComponentTags...>>{}(id);
  }
};
}  // namespace std

// namespace db {
// template <typename TagList, typename Tag>
// struct Subitems<
//     TagList, Tag,
//     Requires<tt::is_a_v<Elliptic::IterationId, item_type<Tag, TagList>>>> {
//   using type = typename item_type<Tag>::tags_list;

//   template <typename Subtag>
//   static void create_item(
//       const gsl::not_null<item_type<Tag>*> parent_value,
//       const gsl::not_null<item_type<Subtag>*> sub_value) noexcept {
//     // TODO: make working both ways
//     *sub_value = get<Subtag>(*parent_value);
//   }

//   template <typename Subtag>
//   static const item_type<Subtag>& create_compute_item(
//       const item_type<Tag>& parent_value) noexcept {
//     return get<Subtag>(parent_value);
//   }
// };
// }  // namespace db
