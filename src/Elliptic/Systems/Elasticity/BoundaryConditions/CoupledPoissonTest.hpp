// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Elasticity::BoundaryConditions {
namespace detail {

template <size_t Dim>
struct CoupledPoissonTestImpl {
  static constexpr Options::String help =
      "Coupling the Poisson equations through boundary conditions.";
  using options = tmpl::list<>;

  CoupledPoissonTestImpl() = default;
  CoupledPoissonTestImpl(const CoupledPoissonTestImpl&) noexcept = default;
  CoupledPoissonTestImpl& operator=(const CoupledPoissonTestImpl&) noexcept =
      default;
  CoupledPoissonTestImpl(CoupledPoissonTestImpl&&) noexcept = default;
  CoupledPoissonTestImpl& operator=(CoupledPoissonTestImpl&&) noexcept =
      default;
  ~CoupledPoissonTestImpl() noexcept = default;

  using argument_tags =
      tmpl::list<domain::Tags::Direction<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::Normalized<
                     domain::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>,
                 domain::Tags::Mesh<Dim>, ::Tags::AnalyticSolutionsBase>;
  using volume_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, ::Tags::AnalyticSolutionsBase>;

  void apply(
      gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
      gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress,
      const Direction<Dim>& direction, const tnsr::I<DataVector, Dim>& x,
      const tnsr::i<DataVector, Dim>& face_normal, const Mesh<Dim>& mesh,
      const std::optional<
          std::reference_wrapper<const Variables<db::wrap_tags_in<
              ::Tags::Analytic,
              tmpl::list<Tags::Displacement<Dim>, Tags::MinusStress<Dim>>>>>>
          analytic_solutions) const noexcept;

  using argument_tags_linearized = tmpl::list<domain::Tags::Direction<Dim>>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<tnsr::I<DataVector, Dim>*> displacement_correction,
      gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress_correction,
      const Direction<Dim>& direction) noexcept;

  // NOLINTNEXTLINE
  void pup(PUP::er& /*p*/) noexcept {}
};

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <size_t Dim, typename Registrars>
struct CoupledPoissonTest;

namespace Registrars {
template <size_t Dim>
struct CoupledPoissonTest {
  template <typename Registrars>
  using f = BoundaryConditions::CoupledPoissonTest<Dim, Registrars>;
};
}  // namespace Registrars
/// \endcond

template <size_t Dim,
          typename Registrars = tmpl::list<Registrars::CoupledPoissonTest<Dim>>>
class CoupledPoissonTest
    : public elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>,
      public detail::CoupledPoissonTestImpl<Dim> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>;

 public:
  CoupledPoissonTest() = default;
  CoupledPoissonTest(const CoupledPoissonTest&) noexcept = default;
  CoupledPoissonTest& operator=(const CoupledPoissonTest&) noexcept = default;
  CoupledPoissonTest(CoupledPoissonTest&&) noexcept = default;
  CoupledPoissonTest& operator=(CoupledPoissonTest&&) noexcept = default;
  ~CoupledPoissonTest() noexcept = default;

  /// \cond
  explicit CoupledPoissonTest(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CoupledPoissonTest);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const noexcept override {
    return std::make_unique<CoupledPoissonTest>(*this);
  }

  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    detail::CoupledPoissonTestImpl<Dim>::pup(p);
  }
};

/// \cond
template <size_t Dim, typename Registrars>
PUP::able::PUP_ID CoupledPoissonTest<Dim, Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Elasticity::BoundaryConditions
