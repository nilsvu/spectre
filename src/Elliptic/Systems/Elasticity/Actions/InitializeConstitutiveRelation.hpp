// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Elasticity {

namespace OptionTags {
template <size_t Dim>
struct ConstitutiveRelationPerBlock {
  static std::string name() { return "Material"; }
  using ConstRelPtr =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
  using type = std::variant<ConstRelPtr, std::vector<ConstRelPtr>,
                                 std::unordered_map<std::string, ConstRelPtr>>;
  static constexpr Options::String help =
      "A constitutive relation in every block of the domain.";
};
}  // namespace OptionTags

namespace Tags {
struct ConstitutiveRelationPerBlockBase : db::BaseTag {};

/// A constitutive relation in every block of the domain. Either constructed
/// from input-file options or retrieved from the analytic solution, if one
/// exists.
template <size_t Dim>
struct ConstitutiveRelationPerBlock : db::SimpleTag,
                                      ConstitutiveRelationPerBlockBase {
  using ConstRelPtr =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
  using type = std::vector<ConstRelPtr>;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>,
                             OptionTags::ConstitutiveRelationPerBlock<Dim>>;
  static type create_from_options(
      const std::unique_ptr<DomainCreator<Dim>>& domain_creator,
      const std::variant<ConstRelPtr, std::vector<ConstRelPtr>,
                       std::unordered_map<std::string, ConstRelPtr>>&
          option_value) {
    const auto block_names = domain_creator->block_names();
    const auto block_groups = domain_creator->block_groups();
    const domain::ExpandOverBlocks<ConstRelPtr> expand_over_blocks{
        block_names, block_groups};
    try {
      return std::visit(expand_over_blocks, option_value);
    } catch (const std::exception& error) {
      ERROR("Invalid 'Material': " << error.what());
    }
  }
};

/// References the constitutive relation for the element's block, which is
/// stored in the global cache
template <size_t Dim>
struct ConstitutiveRelationReference : ConstitutiveRelation<Dim>,
                                       db::ReferenceTag {
  using base = ConstitutiveRelation<Dim>;
  using argument_tags = tmpl::list<ConstitutiveRelationPerBlockBase,
                                   domain::Tags::Element<Dim>>;
  static const ConstitutiveRelations::ConstitutiveRelation<Dim>& get(
      const std::vector<
          std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>>&
          constitutive_relation_per_block,
      const Element<Dim>& element) {
    return *constitutive_relation_per_block.at(element.id().block_id());
  }
};
}  // namespace Tags

/// Actions related to solving Elasticity systems
namespace Actions {

/*!
 * \brief Initialize the constitutive relation describing properties of the
 * elastic material
 *
 * Every block in the domain can have a different constitutive relation,
 * allowing for composite materials. All constitutive relations are stored in
 * the global cache indexed by block, and elements reference their block's
 * constitutive relation in the DataBox. This means an element can retrieve the
 * local constitutive relation from the DataBox simply by requesting
 * `Elasticity::Tags::ConstitutiveRelation<Dim>`.
 */
template <size_t Dim>
struct InitializeConstitutiveRelation {
 private:
 public:
  using const_global_cache_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationPerBlock<Dim>>;
  using simple_tags = tmpl::list<>;
  using compute_tags =
      tmpl::list<Elasticity::Tags::ConstitutiveRelationReference<Dim>>;
  using initialization_tags = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Elasticity
