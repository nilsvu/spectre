// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <cstddef>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Protocols.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainPreconditioners/ExplicitInverse.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct SchwarzSmoother {
  static constexpr Options::String help =
      "Options for the iterative Schwarz smoother";
};

// [subdomain_operator]
// Applies the linear operator given explicitly in the input file to data on an
// element-centered subdomain, using standard matrix multiplications.
//
// We assume all elements have the same extents so we can use the element's
// intruding overlap extents instead of initializing extruding overlap extents.
struct SubdomainElementOperator {
  using argument_tags = tmpl::list<
      helpers_distributed::LinearOperator, domain::Tags::Element<1>,
      LinearSolver::Schwarz::Tags::IntrudingExtents<1, SchwarzSmoother>>;

  template <typename SubdomainData, typename ResultData,
            typename SubdomainOperator>
  static void apply(
      const db::item_type<helpers_distributed::LinearOperator>& linear_operator,
      const Element<1>& element,
      const std::array<size_t, 1>& all_intruding_extents,
      const SubdomainData& subdomain_data,
      const gsl::not_null<ResultData*> result,
      const gsl::not_null<SubdomainOperator*> /*subdomain_operator*/) noexcept {
    const size_t element_index = helpers_distributed::get_index(element.id());

    // Get the operator matrix slice for the central element
    const auto& operator_slice = gsl::at(linear_operator, element_index);
    const size_t num_points = operator_slice.columns();
    const Index<1> volume_extents{num_points};

    // Apply contribution to the central element
    const Matrix operator_matrix_element = blaze::submatrix(
        operator_slice, element_index * num_points, 0, num_points, num_points);
    apply_matrices(make_not_null(&result->element_data),
                   make_array<1>(operator_matrix_element),
                   subdomain_data.element_data, volume_extents);

    // Apply contribution to the overlaps
    for (const auto& [overlap_id, overlap_data] : subdomain_data.overlap_data) {
      // Silence unused-variable warning on GCC 7
      (void)overlap_data;
      const auto& direction = overlap_id.first;
      const size_t overlapped_element_index = direction.side() == Side::Lower
                                                  ? (element_index - 1)
                                                  : (element_index + 1);
      const Matrix operator_matrix_overlap = blaze::submatrix(
          operator_slice, overlapped_element_index * num_points, 0, num_points,
          num_points);
      const auto overlap_contribution_extended =
          apply_matrices(make_array<1>(operator_matrix_overlap),
                         subdomain_data.element_data, volume_extents);
      const auto& overlap_extents =
          gsl::at(all_intruding_extents, direction.dimension());
      const auto direction_from_neighbor = direction.opposite();
      LinearSolver::Schwarz::data_on_overlap(
          make_not_null(&result->overlap_data[overlap_id]),
          overlap_contribution_extended, volume_extents, overlap_extents,
          direction_from_neighbor);
    }
  }
};

template <typename Directions>
struct SubdomainFaceOperator;

template <>
struct SubdomainFaceOperator<domain::Tags::InternalDirections<1>> {
  using argument_tags = tmpl::list<
      helpers_distributed::LinearOperator, domain::Tags::Element<1>,
      domain::Tags::Direction<1>,
      LinearSolver::Schwarz::Tags::IntrudingExtents<1, SchwarzSmoother>>;
  using volume_tags = tmpl::list<
      helpers_distributed::LinearOperator, domain::Tags::Element<1>,
      LinearSolver::Schwarz::Tags::IntrudingExtents<1, SchwarzSmoother>>;

  template <typename SubdomainData, typename ResultData,
            typename SubdomainOperator>
  static void apply(
      const db::item_type<helpers_distributed::LinearOperator>& linear_operator,
      const Element<1>& element, const Direction<1>& direction,
      const std::array<size_t, 1>& all_intruding_extents,
      const SubdomainData& subdomain_data,
      const gsl::not_null<ResultData*> result,
      const gsl::not_null<SubdomainOperator*> /*subdomain_operator*/) noexcept {
    const size_t element_index = helpers_distributed::get_index(element.id());
    for (const auto& [overlap_id, overlap_data] : subdomain_data.overlap_data) {
      const auto& local_direction = overlap_id.first;
      // Apply the operator only to this face in this invocation
      if (local_direction != direction) {
        continue;
      }
      const auto direction_from_neighbor = direction.opposite();
      const size_t overlapped_element_index = direction.side() == Side::Lower
                                                  ? (element_index - 1)
                                                  : (element_index + 1);
      const auto& overlap_extents =
          gsl::at(all_intruding_extents, direction.dimension());

      // Get the operator matrix slice for the overlapped element
      const auto& operator_slice =
          gsl::at(linear_operator, overlapped_element_index);
      const size_t num_points = operator_slice.columns();
      const Index<1> volume_extents{num_points};

      // Get the overlap data extended by zeros
      const auto extended_overlap_data =
          LinearSolver::Schwarz::extended_overlap_data(
              overlap_data, volume_extents, overlap_extents,
              direction_from_neighbor);

      // Apply contribution to the central element
      const Matrix operator_matrix_element =
          blaze::submatrix(operator_slice, element_index * num_points, 0,
                           num_points, num_points);
      result->element_data +=
          apply_matrices(make_array<1>(operator_matrix_element),
                         extended_overlap_data, volume_extents);

      // Apply contribution to the overlap
      const Matrix operator_matrix_overlap = blaze::submatrix(
          operator_slice, overlapped_element_index * num_points, 0, num_points,
          num_points);
      const auto overlap_contribution_extended =
          apply_matrices(make_array<1>(operator_matrix_overlap),
                         extended_overlap_data, volume_extents);
      result->overlap_data.at(overlap_id) +=
          LinearSolver::Schwarz::data_on_overlap(
              overlap_contribution_extended, volume_extents, overlap_extents,
              direction_from_neighbor);
    }
  }
};

template <>
struct SubdomainFaceOperator<domain::Tags::BoundaryDirectionsInterior<1>> {
  using argument_tags = tmpl::list<>;
  template <typename SubdomainData, typename ResultData,
            typename SubdomainOperator>
  static void apply(
      const SubdomainData& /*subdomain_data*/,
      const gsl::not_null<ResultData*> /*result*/,
      const gsl::not_null<SubdomainOperator*> /*subdomain_operator*/) noexcept {
  }
};

struct SubdomainOperator
    : tt::ConformsTo<LinearSolver::Schwarz::protocols::SubdomainOperator> {
  static constexpr size_t volume_dim = 1;
  using element_operator = SubdomainElementOperator;
  template <typename Directions>
  using face_operator = SubdomainFaceOperator<Directions>;
  explicit SubdomainOperator(const size_t /*element_num_points*/) noexcept {}
};
// [subdomain_operator]

static_assert(tt::assert_conforms_to<
              SubdomainOperator,
              LinearSolver::Schwarz::protocols::SubdomainOperator>);

struct Metavariables {
  static constexpr const char* const help{
      "Test the Schwarz linear solver algorithm"};
  static constexpr size_t volume_dim = 1;

  using subdomain_preconditioner =
      LinearSolver::Schwarz::subdomain_preconditioners::ExplicitInverse<1>;
  using linear_solver =
      LinearSolver::Schwarz::Schwarz<helpers_distributed::fields_tag,
                                     SchwarzSmoother, SubdomainOperator,
                                     subdomain_preconditioner>;
  using preconditioner = void;

  using Phase = helpers::Phase;
  using observed_reduction_data_tags =
      helpers::observed_reduction_data_tags<Metavariables>;
  using component_list = helpers_distributed::component_list<Metavariables>;
  static constexpr bool ignore_unrecognized_command_line_options = false;
  static constexpr auto determine_next_phase =
      helpers::determine_next_phase<Metavariables>;
};

}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};

using charmxx_main_component = Parallel::Main<Metavariables>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
