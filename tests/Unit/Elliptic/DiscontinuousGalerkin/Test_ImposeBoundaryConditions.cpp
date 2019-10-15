// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <boost/functional/hash/extensions.hpp>

// IWYU pragma: no_include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare dg::Actions::ReceiveDataForFluxes

// IWYU pragma: no_include <boost/variant/get.hpp>

// Note: Most of this test is adapted from:
// `NumericalAlgorithms/DiscontinuousGalerkin/Actions/
// Test_ImposeBoundaryConditions.cpp`

namespace {
constexpr size_t Dim = 2;

struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarField"; }
  using type = Scalar<DataVector>;
};

using field_tag = ScalarFieldTag;
using vars_tag = Tags::Variables<tmpl::list<field_tag>>;

struct System {
  static constexpr const size_t volume_dim = Dim;
  using variables_tag = vars_tag;
  using primal_variables = tmpl::list<ScalarFieldTag>;
};

using interior_bdry_vars_tag =
    ::Tags::Interface<::Tags::BoundaryDirectionsInterior<Dim>, vars_tag>;
using exterior_bdry_vars_tag =
    ::Tags::Interface<::Tags::BoundaryDirectionsExterior<Dim>, vars_tag>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;

  using simple_tags =
      db::AddSimpleTags<interior_bdry_vars_tag, exterior_bdry_vars_tag>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::dg::Actions::
                         ImposeHomogeneousDirichletBoundaryConditions<
                             Metavariables>>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Actions.BoundaryConditions",
                  "[Unit][Elliptic][Actions]") {
  using my_component = ElementArray<Metavariables>;
  // Just making up two "external" directions
  const auto external_directions = {Direction<2>::lower_eta(),
                                    Direction<2>::upper_xi()};
  const size_t num_points = 3;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  {
    db::item_type<interior_bdry_vars_tag> interior_bdry_vars;
    db::item_type<exterior_bdry_vars_tag> exterior_bdry_vars;
    for (const auto& direction : external_directions) {
      interior_bdry_vars[direction].initialize(3);
      exterior_bdry_vars[direction].initialize(3);
    }
    get<field_tag>(interior_bdry_vars[Direction<2>::lower_eta()]) =
        Scalar<DataVector>{num_points, 1.};
    get<field_tag>(interior_bdry_vars[Direction<2>::upper_xi()]) =
        Scalar<DataVector>{num_points, 2.};

    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, 0,
        {std::move(interior_bdry_vars), std::move(exterior_bdry_vars)});
  }
  runner.set_phase(Metavariables::Phase::Testing);

  ActionTesting::next_action<my_component>(make_not_null(&runner), 0);

  // Check that BC's were indeed applied.
  const auto& exterior_vars =
      ActionTesting::get_databox_tag<my_component, exterior_bdry_vars_tag>(
          runner, 0);
  db::item_type<exterior_bdry_vars_tag> expected_vars{};
  for (const auto& direction : external_directions) {
    expected_vars[direction].initialize(3);
  }
  get<field_tag>(expected_vars[Direction<2>::lower_eta()]) =
      Scalar<DataVector>{num_points, -1.};
  get<field_tag>(expected_vars[Direction<2>::upper_xi()]) =
      Scalar<DataVector>{num_points, -2.};
  CHECK(exterior_vars == expected_vars);
}

}  // namespace
