// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {

struct ScalarField : db::SimpleTag {
  static std::string name() noexcept { return "ScalarField"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<ScalarField> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<ScalarField> /*meta*/) const noexcept {
    return {Scalar<DataVector>(2. * get<0>(x))};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarField>>;
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<::Tags::Coordinates<Dim, Frame::Inertial>>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Elliptic::Actions::InitializeAnalyticSolution>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<AnalyticSolution<Dim>>;
  using const_global_cache_tag_list = tmpl::list<analytic_solution_tag>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeAnalyticSolution",
                  "[Unit][Elliptic][Actions]") {
  const tnsr::I<DataVector, 1, Frame::Inertial> inertial_coords{
      {{{1., 2., 3., 4.}}}};

  const ElementId<1> element_id{0};
  using metavariables = Metavariables<1>;
  using element_array = ElementArray<1, metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {AnalyticSolution<1>{}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id, {inertial_coords});
  runner.set_phase(metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  const Scalar<DataVector> expected_analytic_solution{{{{2., 4., 6., 8.}}}};
  CHECK_ITERABLE_APPROX(get(get_tag(::Tags::Analytic<ScalarField>{})),
                        get(expected_analytic_solution));
}
