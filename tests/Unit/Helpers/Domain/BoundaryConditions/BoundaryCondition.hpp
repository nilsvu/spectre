// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TestingFrameworkGroup
/// \brief Helpers for boundary conditions
namespace TestHelpers::domain::BoundaryConditions {
/// \cond
template <size_t Dim>
class TestBoundaryCondition;
/// \endcond

/// \brief A system-specific boundary condition base class.
///
/// To be used in conjunction with `SystemWithBoundaryConditions`
template <size_t Dim>
class BoundaryConditionBase
    : public ::domain::BoundaryConditions::BoundaryCondition {
 public:
  using creatable_classes = tmpl::list<
      TestBoundaryCondition<Dim>,
      ::domain::BoundaryConditions::None<BoundaryConditionBase<Dim>>,
      ::domain::BoundaryConditions::Periodic<BoundaryConditionBase<Dim>>>;

  BoundaryConditionBase() = default;
  BoundaryConditionBase(BoundaryConditionBase&&) = default;
  BoundaryConditionBase& operator=(BoundaryConditionBase&&) = default;
  BoundaryConditionBase(const BoundaryConditionBase&) = default;
  BoundaryConditionBase& operator=(const BoundaryConditionBase&) = default;
  ~BoundaryConditionBase() override = default;

  explicit BoundaryConditionBase(CkMigrateMessage* msg);

  void pup(PUP::er& p) override;

  static constexpr Options::String help = {"Boundary conditions for tests."};
};

/// \brief Concrete boundary condition
template <size_t Dim>
class TestBoundaryCondition final : public BoundaryConditionBase<Dim> {
 public:
  TestBoundaryCondition() = default;
  explicit TestBoundaryCondition(Direction<Dim> direction, size_t block_id = 0);
  TestBoundaryCondition(const std::string& direction, size_t block_id);
  TestBoundaryCondition(TestBoundaryCondition&&) = default;
  TestBoundaryCondition& operator=(TestBoundaryCondition&&) = default;
  TestBoundaryCondition(const TestBoundaryCondition&) = default;
  TestBoundaryCondition& operator=(const TestBoundaryCondition&) = default;
  ~TestBoundaryCondition() override = default;
  explicit TestBoundaryCondition(CkMigrateMessage* const msg);

  struct DirectionOptionTag {
    using type = std::string;
    static std::string name() { return "Direction"; }
    static constexpr Options::String help =
        "The direction the boundary condition operates in.";
  };
  struct BlockIdOptionTag {
    using type = size_t;
    static std::string name() { return "BlockId"; }
    static constexpr Options::String help =
        "The id of the block the boundary condition operates in.";
  };

  using options = tmpl::list<DirectionOptionTag, BlockIdOptionTag>;

  static constexpr Options::String help = {"Boundary condition for testing."};

  WRAPPED_PUPable_decl_base_template(
      ::domain::BoundaryConditions::BoundaryCondition,
      TestBoundaryCondition<Dim>);

  const Direction<Dim>& direction() const { return direction_; }
  size_t block_id() const { return block_id_; }

  auto get_clone() const -> std::unique_ptr<
      ::domain::BoundaryConditions::BoundaryCondition> override;

  void pup(PUP::er& p) override;

 private:
  Direction<Dim> direction_{};
  size_t block_id_{0};
};

template <size_t Dim>
bool operator==(const TestBoundaryCondition<Dim>& lhs,
                const TestBoundaryCondition<Dim>& rhs);

template <size_t Dim>
bool operator!=(const TestBoundaryCondition<Dim>& lhs,
                const TestBoundaryCondition<Dim>& rhs);

template <size_t Dim>
using TestPeriodicBoundaryCondition =
    ::domain::BoundaryConditions::Periodic<BoundaryConditionBase<Dim>>;

template <size_t Dim>
using TestNoneBoundaryCondition =
    ::domain::BoundaryConditions::None<BoundaryConditionBase<Dim>>;

/// Empty system that has boundary conditions
template <size_t Dim>
struct SystemWithBoundaryConditions {
  using boundary_conditions_base = BoundaryConditionBase<Dim>;
};

/// Empty system that doesn't have boundary conditions
template <size_t Dim>
struct SystemWithoutBoundaryConditions {};

/// Metavariables with a system that has boundary conditions
template <size_t Dim, typename Creator>
struct MetavariablesWithBoundaryConditions {
  using system = SystemWithBoundaryConditions<Dim>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DomainCreator<Dim>, tmpl::list<Creator>>>;
  };
};

/// Metavariables with a system that doesn't have boundary conditions
template <size_t Dim, typename Creator>
struct MetavariablesWithoutBoundaryConditions {
  using system = SystemWithoutBoundaryConditions<Dim>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DomainCreator<Dim>, tmpl::list<Creator>>>;
  };
};

/// Assuming `all_boundary_conditions` are of type `TestBoundaryCondition`,
/// check their direction and block ID
template <size_t Dim>
void test_boundary_conditions(
    const std::vector<DirectionMap<
        Dim, std::unique_ptr<::domain::BoundaryConditions::BoundaryCondition>>>&
        all_boundary_conditions);

/// Helper function to factory-create a domain creator in tests with or without
/// boundary conditions
template <size_t Dim, typename Creator>
std::unique_ptr<::DomainCreator<Dim>> test_creation(
    const std::string& option_string, const bool with_boundary_conditions) {
  auto created = [&option_string, &with_boundary_conditions]() {
    if (with_boundary_conditions) {
      return TestHelpers::test_option_tag<
          ::domain::OptionTags::DomainCreator<Dim>,
          MetavariablesWithBoundaryConditions<Dim, Creator>>(option_string);
    } else {
      return TestHelpers::test_option_tag<
          ::domain::OptionTags::DomainCreator<Dim>,
          MetavariablesWithoutBoundaryConditions<Dim, Creator>>(option_string);
    }
  }();
  REQUIRE(dynamic_cast<const Creator*>(created.get()) != nullptr);
  return created;
}

void register_derived_with_charm();
}  // namespace TestHelpers::domain::BoundaryConditions
