// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include <cstddef>
#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/LinearSolverAlgorithmTestHelpers.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainData.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace helpers = LinearSolverAlgorithmTestHelpers;
namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace {

struct SchwarzSmoother {
  static constexpr OptionString help =
      "Options for the iterative Schwarz smoother";
};

// ---
// TODO: Split overlap data into fields and extra data

template <size_t Dim>
struct OverlapData {
  void orient(const OrientationMap<Dim>& /*orientation*/) noexcept {}
  void pup(PUP::er&) noexcept {}
  OverlapData& operator+=(const OverlapData<Dim>& /*rhs*/) noexcept {
    return *this;
  }
  OverlapData& operator-=(const OverlapData<Dim>& /*rhs*/) noexcept {
    return *this;
  }
  OverlapData& operator/=(const double /*scalar*/) noexcept { return *this; }
  template <typename FieldTags>
  void add_to(const gsl::not_null<Variables<FieldTags>*> /*lhs*/) const
      noexcept {}
};

template <size_t Dim>
OverlapData<Dim> operator-(const OverlapData<Dim>& /*lhs*/,
                           const OverlapData<Dim>& /*rhs*/) noexcept {
  return {};
}

template <size_t Dim>
OverlapData<Dim> operator*(const double, const OverlapData<Dim>&)noexcept {
  return {};
}
}  // namespace

namespace LinearSolver {
namespace InnerProductImpls {

template <size_t Dim>
struct InnerProductImpl<OverlapData<Dim>, OverlapData<Dim>> {
  static double apply(const OverlapData<Dim>&,
                      const OverlapData<Dim>&) noexcept {
    return 0.;
  }
};

}  // namespace InnerProductImpls
}  // namespace LinearSolver

// ---

namespace {

template <size_t Dim>
struct CollectOverlapData {
  using argument_tags = tmpl::list<>;
  template <typename FieldsType>
  auto operator()(const FieldsType& /*fields*/) const noexcept {
    return OverlapData<Dim>{};
  }
};

template <size_t Dim>
struct SubdomainOperator {
  static constexpr size_t volume_dim = Dim;
  using SubdomainDataType = LinearSolver::schwarz_detail::SubdomainData<
      volume_dim, db::item_type<helpers_distributed::fields_tag>,
      OverlapData<volume_dim>>;
  using collect_overlap_data = CollectOverlapData<Dim>;

  explicit SubdomainOperator(const size_t central_num_points) noexcept
      : result_{central_num_points} {}

  const SubdomainDataType& result() const noexcept { return result_; }

  struct volume_operator {
    using argument_tags = tmpl::list<helpers_distributed::LinearOperator,
                                     domain::Tags::Element<1>>;

    static void apply(
        const db::item_type<helpers_distributed::LinearOperator>&
            linear_operator,
        const Element<1>& element, const SubdomainDataType& arg,
        const gsl::not_null<SubdomainOperator*> subdomain_operator) noexcept {
      size_t array_index = element.id().segment_ids()[0].index();
      // Parallel::printf("Applying operator on %d...\n", array_index);
      const auto& operator_slice = gsl::at(linear_operator, array_index);
      const size_t num_points = operator_slice.columns();
      const DenseMatrix<double, blaze::columnMajor> subdomain_operator_matrix =
          blaze::submatrix(operator_slice, array_index * num_points, 0,
                           num_points, num_points);
      // Apply matrix to central element data
      dgemv_('N', num_points, num_points, 1, subdomain_operator_matrix.data(),
             num_points, arg.element_data.data(), 1, 0,
             subdomain_operator->result_.element_data.data(), 1);
      // TODO: Add boundary contributions
      // Parallel::printf("%d operand: %s\n", array_index, arg.element_data);
      // Parallel::printf("%d operator: %s\n", array_index, subdomain_operator);
      // Parallel::printf("%d applied: %s\n", array_index, result.element_data);
      subdomain_operator->result_.boundary_data = arg.boundary_data;
    }
  };

  struct face_operator {
    using argument_tags = tmpl::list<>;

    int operator()(
        const SubdomainDataType& /*arg*/,
        const gsl::not_null<SubdomainOperator*> /*subdomain_operator*/) const
        noexcept {
      return 0;
    }
  };

  SubdomainDataType result_;
};

struct WeightingOperator {
  using argument_tags = tmpl::list<>;
  template <typename SubdomainOperatorType>
  static void apply(
      const gsl::not_null<SubdomainOperatorType*> /*subdomain_data*/) {}
};

template <size_t Dim>
struct Metavariables {
  static constexpr const char* const help{
      "Test the Schwarz linear solver algorithm"};

  static constexpr size_t volume_dim = Dim;

  using linear_solver =
      LinearSolver::Schwarz<Metavariables, helpers_distributed::fields_tag,
                            SchwarzSmoother, SubdomainOperator<Dim>,
                            WeightingOperator>;
  using preconditioner = void;

  using Phase = helpers::Phase;
  using element_observation_type = helpers::element_observation_type;
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

using charmxx_main_component = Parallel::Main<Metavariables<1>>;

#include "Parallel/CharmMain.tpp"  // IWYU pragma: keep
