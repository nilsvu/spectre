// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/container_hash/hash.hpp>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>

#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/ExplicitInverse.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/LinearSolver.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// Linear solvers that approximately invert the
/// `elliptic::dg::subdomain_operator::SubdomainOperator` to make the Schwarz
/// subdomain solver converge faster.
/// \see LinearSolver::Schwarz::Schwarz
namespace elliptic::subdomain_preconditioners {

/// \cond
template <typename PoissonSystem, typename OptionsGroup, typename Solver,
          typename LinearSolverRegistrars>
struct MinusLaplacian;
/// \endcond

namespace Registrars {
template <
    typename PoissonSystem, typename OptionsGroup,
    typename Solver = LinearSolver::Serial::LinearSolver<tmpl::list<
        ::LinearSolver::Serial::Registrars::Gmres<
            ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
                PoissonSystem::volume_dim, tmpl::list<Poisson::Tags::Field>>>,
        ::LinearSolver::Serial::Registrars::ExplicitInverse>>>
struct MinusLaplacian {
  template <typename LinearSolverRegistrars>
  using f =
      subdomain_preconditioners::MinusLaplacian<PoissonSystem, OptionsGroup,
                                                Solver, LinearSolverRegistrars>;
};
}  // namespace Registrars

/*!
 * \brief Approximate the subdomain operator with a flat-space Laplacian with
 * Dirichlet boundary conditions for every tensor component separately.
 *
 * This linear solver applies the `Solver` to every tensor component in
 * turn, approximating the subdomain operator with a flat-space Laplacian with
 * Dirichlet boundary conditions. This can be a lot cheaper than solving the
 * full subdomain operator and can provide effective preconditioning for an
 * iterative subdomain solver. The approximation is better the closer the
 * original PDEs are to a set of decoupled flat-space Poisson equations with
 * Dirichlet boundary conditions.
 *
 * \tparam PoissonSystem Poisson system
 * \tparam OptionsGroup The options group identifying the
 * `LinearSolver::Schwarz::Schwarz` solver that defines the subdomain geometry.
 * \tparam Solver Any class that provides a `solve` and a `reset` function,
 * but typically a `LinearSolver::Serial::LinearSolver`. The solver will be
 * factory-created from input-file options.
 */
template <
    typename PoissonSystem, typename OptionsGroup,
    typename Solver = LinearSolver::Serial::LinearSolver<tmpl::list<
        ::LinearSolver::Serial::Registrars::Gmres<
            ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
                PoissonSystem::volume_dim, tmpl::list<Poisson::Tags::Field>>>,
        ::LinearSolver::Serial::Registrars::ExplicitInverse>>,
    typename LinearSolverRegistrars = tmpl::list<
        Registrars::MinusLaplacian<PoissonSystem, OptionsGroup, Solver>>>
class MinusLaplacian
    : public LinearSolver::Serial::LinearSolver<LinearSolverRegistrars> {
 private:
  static constexpr size_t Dim = PoissonSystem::volume_dim;
  using Base = LinearSolver::Serial::LinearSolver<LinearSolverRegistrars>;
  using StoredSolverType = tmpl::conditional_t<std::is_abstract_v<Solver>,
                                               std::unique_ptr<Solver>, Solver>;
  // Identifies an external block boundary, for boundary conditions
  using BoundaryId = std::pair<size_t, Direction<Dim>>;

 public:
  static constexpr size_t volume_dim = Dim;
  using options_group = OptionsGroup;
  using poisson_system = PoissonSystem;
  using BoundaryConditionsBase =
      typename poisson_system::boundary_conditions_base;
  using SubdomainOperator =
      elliptic::dg::subdomain_operator::SubdomainOperator<poisson_system,
                                                          OptionsGroup>;
  using SubdomainData = ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
      Dim, tmpl::list<Poisson::Tags::Field>>;
  // Associates Dirichlet or Neumann conditions to every external block
  // boundary. For every configuration of this type we'll need a separate
  // solver, which may cache the Poisson operator matrix that imposes the
  // corresponding boundary conditions. This type is used as key in other maps,
  // so it must be hashable.
  using BoundaryConditionsSignature =
      std::map<BoundaryId, elliptic::BoundaryConditionType>;
  using solver_type = Solver;

  struct SolverOptionTag {
    static std::string name() { return "Solver"; }
    using type = StoredSolverType;
    static constexpr Options::String help =
        "The linear solver used to invert the Laplace operator. The solver is "
        "shared between tensor components with the same type of boundary "
        "conditions (Dirichlet-type or Neumann-type).";
  };

  struct BoundaryConditions {
    using type = Options::Auto<elliptic::BoundaryConditionType>;
    static constexpr Options::String help =
        "The boundary conditions imposed by the Laplace operator. Specify "
        "'Auto' to choose between homogeneous Dirichlet or Neumann boundary "
        "conditions automatically, based on the configuration of the the full "
        "operator.";
  };

  using options = tmpl::list<SolverOptionTag, BoundaryConditions>;
  static constexpr Options::String help =
      "Approximate the linear operator with a Laplace operator with Dirichlet "
      "or Neumann boundary conditions for every tensor component separately.";

  MinusLaplacian() = default;
  MinusLaplacian(MinusLaplacian&& /*rhs*/) = default;
  MinusLaplacian& operator=(MinusLaplacian&& /*rhs*/) = default;
  ~MinusLaplacian() = default;
  MinusLaplacian(const MinusLaplacian& rhs)
      : Base(rhs),
        solver_(rhs.clone_solver()),
        boundary_condition_type_(rhs.boundary_condition_type_) {}
  MinusLaplacian& operator=(const MinusLaplacian& rhs) {
    Base::operator=(rhs);
    solver_ = rhs.clone_solver();
    boundary_condition_type_ = rhs.boundary_condition_type_;
    return *this;
  }

  /// \cond
  explicit MinusLaplacian(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(MinusLaplacian);  // NOLINT
  /// \endcond

  MinusLaplacian(
      StoredSolverType solver,
      std::optional<elliptic::BoundaryConditionType> boundary_condition_type)
      : solver_(std::move(solver)),
        boundary_condition_type_(boundary_condition_type) {}

  const Solver& solver() const {
    if constexpr (std::is_abstract_v<Solver>) {
      return *solver_;
    } else {
      return solver_;
    }
  }

  /// Solve the equation \f$Ax=b\f$ by approximating \f$A\f$ with a Laplace
  /// operator with homogeneous Dirichlet or Neumann boundary conditions for
  /// every tensor component in \f$x\f$.
  template <typename System, typename VarsType, typename SourceType,
            typename... SubdomainOperatorParams, typename... OperatorArgs>
  Convergence::HasConverged solve(
      gsl::not_null<VarsType*> solution,
      const elliptic::dg::subdomain_operator::SubdomainOperator<
          System, OptionsGroup, SubdomainOperatorParams...>& subdomain_operator,
      const SourceType& source,
      const std::tuple<OperatorArgs...>& operator_args) const;

  void reset() override {
    mutable_solver().reset();
    // These buffers depend on the operator being solved, so they are reset when
    // the operator changes. The other buffers only hold workspace memory so
    // they don't need to be reset.
    bc_signatures_ = std::nullopt;
    solvers_.clear();
    boundary_conditions_.clear();
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Base::pup(p);
    // Serialize original data
    p | solver_;
    p | boundary_condition_type_;
    // Serialize caches that are possibly expensive to re-create
    p | bc_signatures_;
    p | solvers_;
    // Other properties are memory buffers that don't need to be serialized.
  }

  std::unique_ptr<Base> get_clone() const override {
    return std::make_unique<MinusLaplacian>(*this);
  }

 private:
  // For each tensor component, get the set of boundary conditions that should
  // be imposed by the Laplacian operator on that component, based on the domain
  // configuration. These are cached for successive solves of the same operator.
  // An empty list means the subdomain has no external boundaries.
  template <typename System, typename DbTagsList>
  const std::vector<BoundaryConditionsSignature>&
  get_cached_boundary_conditions_signatures(
      const db::DataBox<DbTagsList>& box) const {
    if (not bc_signatures_.has_value()) {
      bc_signatures_ = std::vector<BoundaryConditionsSignature>{};
      static constexpr size_t num_components = Variables<
          typename System::primal_fields>::number_of_independent_components;
      const auto& blocks = db::get<domain::Tags::Domain<Dim>>(box).blocks();
      const auto collect_bc_signatures = [&bc_signatures = *bc_signatures_,
                                          &override_boundary_condition_type =
                                              boundary_condition_type_,
                                          &blocks](
                                             const BoundaryId& boundary_id) {
        if (bc_signatures.empty()) {
          bc_signatures.resize(num_components);
        }
        if (bc_signatures[0].find(boundary_id) != bc_signatures[0].end()) {
          // We have already handled this block boundary before
          return;
        }
        // Get the type of boundary condition (Dirichlet or Neumann) for
        // each tensor component from the domain configuration
        const std::vector<elliptic::BoundaryConditionType> bc_types =
            dynamic_cast<const typename System::boundary_conditions_base&>(
                *blocks.at(boundary_id.first)
                     .external_boundary_conditions()
                     .at(boundary_id.second))
                .boundary_condition_types();
        ASSERT(bc_types.size() == num_components,
               "Unexpected number of boundary-condition types. The "
               "boundary condition in block "
                   << boundary_id.first << ", direction " << boundary_id.second
                   << " returned " << bc_types.size() << " items, but expected "
                   << num_components << " (one for each tensor component).");
        for (size_t component = 0; component < num_components; ++component) {
          bc_signatures[component].emplace(
              boundary_id, override_boundary_condition_type.value_or(
                               bc_types.at(component)));
        }
      };
      const auto& element = db::get<domain::Tags::Element<Dim>>(box);
      for (const auto& direction : element.external_boundaries()) {
        collect_bc_signatures({element.id().block_id(), direction});
      }
      const auto& overlap_elements =
          db::get<LinearSolver::Schwarz::Tags::Overlaps<
              domain::Tags::Element<Dim>, Dim, OptionsGroup>>(box);
      for (const auto& [overlap_id, overlap_element] : overlap_elements) {
        for (const auto& direction : overlap_element.external_boundaries()) {
          collect_bc_signatures({overlap_element.id().block_id(), direction});
        }
      }
    }
    return *bc_signatures_;
  }

  // For a tensor component with the given boundary-condition configuration, get
  // the solver and the set of boundary conditions that will be passed to the
  // Poisson subdomain operator. The solver is cached for successive solves of
  // the same operator. This is very important for performance, since the solver
  // may build up an explicit matrix representation that can be applied to all
  // tensor components with the same boundary conditions.
  std::pair<const Solver&,
            const std::unordered_map<BoundaryId, const BoundaryConditionsBase&,
                                     boost::hash<BoundaryId>>&>
  get_cached_solver_and_boundary_conditions(
      const std::vector<BoundaryConditionsSignature>& bc_signatures,
      const size_t component) const {
    if (bc_signatures.empty()) {
      // The subdomain has no external boundaries. We use the original solver.
      return {solver(), {}};
    }
    // Get the cached solver corresponding to this component's
    // boundary-condition configuration
    const auto& bc_signature = bc_signatures.at(component);
    if (solvers_.find(bc_signature) == solvers_.end()) {
      solvers_.emplace(bc_signature, clone_solver());
    }
    const Solver& cached_solver = [this, &bc_signature]() -> const Solver& {
      if constexpr (std::is_abstract_v<Solver>) {
        return *solvers_.at(bc_signature);
      } else {
        return solvers_.at(bc_signature);
      }
    }();
    // Get the cached set of boundary conditions used to override the domain
    // configuration for this component
    if (boundary_conditions_.find(bc_signature) == boundary_conditions_.end()) {
      auto& bc = boundary_conditions_[bc_signature];
      for (const auto& [boundary_id, bc_type] : bc_signature) {
        bc.emplace(boundary_id,
                   bc_type == elliptic::BoundaryConditionType::Dirichlet
                       ? dirichlet_bc
                       : neumann_bc);
      }
    }
    return {cached_solver, boundary_conditions_[bc_signature]};
  }

  Solver& mutable_solver() {
    if constexpr (std::is_abstract_v<Solver>) {
      return *solver_;
    } else {
      return solver_;
    }
  }

  StoredSolverType clone_solver() const {
    if constexpr (std::is_abstract_v<Solver>) {
      return solver_->get_clone();
    } else {
      return solver_;
    }
  }

  // The option-constructed solver for the separate Poisson problems. This
  // instance is cloned for each unique boundary-condition configuration of the
  // tensor components.
  StoredSolverType solver_{};
  std::optional<elliptic::BoundaryConditionType> boundary_condition_type_;

  // These are caches to keep track of the unique boundary-condition
  // configurations. Each boundary-condition configuration has its own solver,
  // because the solver may cache the operator to speed up repeated solves.
  // - The boundary-condition configuration for each tensor component, or an
  //   empty list if the subdomain has no external boundaries, or `std::nullopt`
  //   to signal a clean cache.
  mutable std::optional<std::vector<BoundaryConditionsSignature>>
      bc_signatures_{};
  // - A clone of the `solver_` for each unique boundary-condition configuration
  mutable std::unordered_map<BoundaryConditionsSignature, StoredSolverType,
                             boost::hash<BoundaryConditionsSignature>>
      solvers_{};
  mutable std::unordered_map<
      BoundaryConditionsSignature,
      std::unordered_map<BoundaryId, const BoundaryConditionsBase&,
                         boost::hash<BoundaryId>>,
      boost::hash<BoundaryConditionsSignature>>
      boundary_conditions_{};

  // These are memory buffers that are kept around for repeated solves. Possible
  // optimization: Free this memory at appropriate times, e.g. when the element
  // has completed a full subdomain solve and goes to the background. In some
  // cases the `subdomain_operator_` is never even used again in subsequent
  // subdomain solves because it is cached as a matrix (see
  // LinearSolver::Serial::ExplicitInverse), so we don't need the memory anymore
  // at all.
  mutable SubdomainOperator subdomain_operator_{};
  mutable SubdomainData source_{};
  mutable SubdomainData initial_guess_in_solution_out_{};

  // These boundary condition instances can be re-used for all tensor components
  const Poisson::BoundaryConditions::Robin<
      Dim, typename BoundaryConditionsBase::registrars>
      dirichlet_bc{1., 0., 0.};
  const Poisson::BoundaryConditions::Robin<
      Dim, typename BoundaryConditionsBase::registrars>
      neumann_bc{0., 1., 0.};
};

namespace detail {
template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
void assign_component(
    const gsl::not_null<::LinearSolver::Schwarz::ElementCenteredSubdomainData<
        Dim, LhsTagsList>*>
        lhs,
    const ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
        Dim, RhsTagsList>& rhs,
    const size_t lhs_component, const size_t rhs_component) {
  // Possible optimization: Once we have non-owning Variables we can use a view
  // into the rhs here instead of copying.
  const size_t num_points_element = rhs.element_data.number_of_grid_points();
  for (size_t i = 0; i < num_points_element; ++i) {
    lhs->element_data.data()[lhs_component * num_points_element + i] =
        rhs.element_data.data()[rhs_component * num_points_element + i];
  }
  for (const auto& [overlap_id, rhs_data] : rhs.overlap_data) {
    const size_t num_points_overlap = rhs_data.number_of_grid_points();
    // The random-access operation is relatively slow because it computes a
    // hash, so it's important for performance to avoid repeating it in every
    // iteration of the loop below.
    auto& lhs_vars = lhs->overlap_data[overlap_id];
    for (size_t i = 0; i < num_points_overlap; ++i) {
      lhs_vars.data()[lhs_component * num_points_overlap + i] =
          rhs_data.data()[rhs_component * num_points_overlap + i];
    }
  }
}
}  // namespace detail

template <typename PoissonSystem, typename OptionsGroup, typename Solver,
          typename LinearSolverRegistrars>
template <typename System, typename VarsType, typename SourceType,
          typename... SubdomainOperatorParams, typename... OperatorArgs>
Convergence::HasConverged
MinusLaplacian<PoissonSystem, OptionsGroup, Solver, LinearSolverRegistrars>::
    solve(const gsl::not_null<VarsType*> initial_guess_in_solution_out,
          const elliptic::dg::subdomain_operator::SubdomainOperator<
              System, OptionsGroup,
              SubdomainOperatorParams...>& /*subdomain_operator*/,
          const SourceType& source,
          const std::tuple<OperatorArgs...>& operator_args) const {
  // Solve each component of the source variables in turn, assuming the operator
  // is a Laplacian. For each component we select either homogeneous Dirichlet
  // or Neumann boundary conditions, based on the type of boundary conditions
  // imposed by the full operator.
  static constexpr size_t num_components = Variables<
      typename System::primal_fields>::number_of_independent_components;
  source_.destructive_resize(source);
  initial_guess_in_solution_out_.destructive_resize(source);
  const std::vector<BoundaryConditionsSignature>& bc_signatures =
      get_cached_boundary_conditions_signatures<System>(get<0>(operator_args));
  for (size_t component = 0; component < num_components; ++component) {
    detail::assign_component(make_not_null(&source_), source, 0, component);
    detail::assign_component(make_not_null(&initial_guess_in_solution_out_),
                             *initial_guess_in_solution_out, 0, component);
    const auto& [solver, boundary_conditions] =
        get_cached_solver_and_boundary_conditions(bc_signatures, component);
    solver.solve(make_not_null(&initial_guess_in_solution_out_),
                 subdomain_operator_, source_,
                 std::tuple_cat(operator_args,
                                std::forward_as_tuple(boundary_conditions)));
    detail::assign_component(initial_guess_in_solution_out,
                             initial_guess_in_solution_out_, component, 0);
  }
  return {0, 0};
}

/// \cond
template <typename PoissonSystem, typename OptionsGroup, typename Solver,
          typename LinearSolverRegistrars>
// NOLINTNEXTLINE
PUP::able::PUP_ID MinusLaplacian<PoissonSystem, OptionsGroup, Solver,
                                 LinearSolverRegistrars>::my_PUP_ID = 0;
/// \endcond

}  // namespace elliptic::subdomain_preconditioners
