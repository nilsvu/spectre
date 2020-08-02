// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elasticity::BoundaryConditions {

enum class MirrorSuspension {
  /// The mirror's back is glued to a wall. Dirichlet conditions on the back and
  /// Neumann conditions on the sides.
  AttachedOnBack,
  /// The mirror's sides are fixed, but the back is free. Dirichlet conditions
  /// on the sides and Neumann conditions on the back. This is the setup
  /// implemented in Lovelace2018 (https://arxiv.org/abs/1707.07774).
  AttachedOnSides
};

}  // namespace Elasticity::BoundaryConditions

template <>
struct Options::create_from_yaml<
    Elasticity::BoundaryConditions::MirrorSuspension> {
  template <typename Metavariables>
  static Elasticity::BoundaryConditions::MirrorSuspension create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
Elasticity::BoundaryConditions::MirrorSuspension
Options::create_from_yaml<Elasticity::BoundaryConditions::MirrorSuspension>::
    create<void>(const Options::Option& options);

namespace Elasticity::BoundaryConditions {

elliptic::BoundaryCondition mirror_suspension_boundary_condition_type(
    const MirrorSuspension mirror_suspension,
    const Direction<3>& direction) noexcept;

struct LinearizedLaserBeam;

struct LaserBeam {
 private:
  struct BeamWidth {
    using type = double;
    static constexpr Options::String help{
        "The laser's beam width r_0 with FWHM = 2*sqrt(ln 2)*r_0"};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct MirrorSuspensionOption {
    static std::string name() noexcept { return "MirrorSuspension"; }
    using type = MirrorSuspension;
    static constexpr Options::String help{
        "The way the mirror is suspended, e.g. 'AttachedOnBack' or "
        "'AttachedOnSides'"};
  };

 public:
  using options = tmpl::list<BeamWidth, MirrorSuspensionOption>;
  static constexpr Options::String help{"Laser beam boundary conditions"};

  using Linearization = LinearizedLaserBeam;

  LaserBeam() = default;
  LaserBeam(const LaserBeam&) noexcept = default;
  LaserBeam& operator=(const LaserBeam&) noexcept = default;
  LaserBeam(LaserBeam&&) noexcept = default;
  LaserBeam& operator=(LaserBeam&&) noexcept = default;
  ~LaserBeam() noexcept = default;

  LaserBeam(double beam_width, MirrorSuspension mirror_suspension) noexcept;

  Linearization linearization() const noexcept;

  template <typename Tag>
  elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, 3>& /*x*/, const Direction<3>& direction,
      Tag /*meta*/) const noexcept {
    return mirror_suspension_boundary_condition_type(mirror_suspension_,
                                                     direction);
  }

  using argument_tags =
      tmpl::list<domain::Tags::Direction<3>,
                 domain::Tags::Coordinates<3, Frame::Inertial>>;
  using volume_tags = tmpl::list<>;

  void apply(const gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
             // This is n_i F^{ij} = n_i Y^{ijkl}(x) S_{kl} = -n_i T^{ij}
             const gsl::not_null<tnsr::I<DataVector, 3>*> minus_n_dot_stress,
             const tnsr::i<DataVector, 3>& normal,
             const Direction<3>& direction,
             const tnsr::I<DataVector, 3>& x) const noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  double beam_width_ = std::numeric_limits<double>::signaling_NaN();
  MirrorSuspension mirror_suspension_{};
};

struct LinearizedLaserBeam {
 public:
  LinearizedLaserBeam() = default;
  LinearizedLaserBeam(const LinearizedLaserBeam&) noexcept = default;
  LinearizedLaserBeam& operator=(const LinearizedLaserBeam&) noexcept = default;
  LinearizedLaserBeam(LinearizedLaserBeam&&) noexcept = default;
  LinearizedLaserBeam& operator=(LinearizedLaserBeam&&) noexcept = default;
  ~LinearizedLaserBeam() noexcept = default;

  LinearizedLaserBeam(MirrorSuspension mirror_suspension) noexcept;

  template <typename Tag>
  elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, 3>& /*x*/, const Direction<3>& direction,
      Tag /*meta*/) const noexcept {
    return mirror_suspension_boundary_condition_type(mirror_suspension_,
                                                     direction);
  }

  using argument_tags =
      tmpl::list<domain::Tags::Direction<3>,
                 domain::Tags::Coordinates<3, Frame::Inertial>>;
  using volume_tags = tmpl::list<>;

  // The linearization of the variable-independent conditions is just zero
  void apply(const gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
             const gsl::not_null<tnsr::I<DataVector, 3>*> minus_n_dot_stress,
             const tnsr::i<DataVector, 3>& /*normal*/,
             const Direction<3>& direction,
             const tnsr::I<DataVector, 3>& x) const noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  MirrorSuspension mirror_suspension_{};
};

}  // namespace Elasticity::BoundaryConditions
