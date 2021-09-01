// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Elasticity {
namespace ConstitutiveRelations {

struct CubicCrystal;  // IWYU pragma: keep

template <size_t Dim>
struct IsotropicHomogeneous;  // IWYU pragma: keep
}  // namespace ConstitutiveRelations
}  // namespace Elasticity
/// \endcond

namespace Elasticity {
/*!
 * \brief Constitutive (stress-strain) relations that characterize the elastic
 * properties of a material
 */
namespace ConstitutiveRelations {

/*!
 * \brief Base class for constitutive (stress-strain) relations that
 * characterize the elastic properties of a material
 *
 * \details A constitutive relation, in the context of elasticity, relates the
 * Stress \f$T^{ij}\f$ and Strain \f$S_{ij}=\nabla_{(i}u_{j)}\f$ within an
 * elastic material (see \ref Elasticity). For small stresses it is approximated
 * by the linear relation
 *
 * \f[
 * T^{ij} = -Y^{ijkl}S_{kl}
 * \f]
 *
 * (Eq. 11.17 in \cite ThorneBlandford2017) that is referred to as _Hooke's
 * law_. The constitutive relation in this linear approximation is determined by
 * the elasticity (or _Young's_) tensor \f$Y^{ijkl}=Y^{(ij)(kl)}=Y^{klij}\f$
 * that generalizes a simple proportionality to a three-dimensional and
 * (possibly) anisotropic material.
 *
 * \note We assume a Euclidean metric in Cartesian coordinates here (for now).
 */
template <size_t Dim>
class ConstitutiveRelation : public PUP::able {
 public:
  static constexpr size_t volume_dim = Dim;

  using creatable_classes = tmpl::list<CubicCrystal, IsotropicHomogeneous<Dim>>;

  ConstitutiveRelation() = default;
  ConstitutiveRelation(const ConstitutiveRelation&) = default;
  ConstitutiveRelation& operator=(const ConstitutiveRelation&) = default;
  ConstitutiveRelation(ConstitutiveRelation&&) = default;
  ConstitutiveRelation& operator=(ConstitutiveRelation&&) = default;
  ~ConstitutiveRelation() override = default;

  WRAPPED_PUPable_abstract(ConstitutiveRelation);  // NOLINT

  /// Returns a `std::unique_ptr` pointing to a copy of the
  /// `ConstitutiveRelation`.
  virtual std::unique_ptr<ConstitutiveRelation<Dim>> get_clone()
      const noexcept = 0;

  /// @{
  /// The constitutive relation that characterizes the elastic properties of a
  /// material
  virtual void stress(gsl::not_null<tnsr::II<DataVector, Dim>*> stress,
                      const tnsr::ii<DataVector, Dim>& strain,
                      const tnsr::I<DataVector, Dim>& x) const noexcept = 0;

  // This overload is provided for the situation where the `stress` variable
  // holds a non-symmetric tensor, as is currently the case when it is held in a
  // `Tags::Flux`. The overload can be removed once it is no longer used.
  void stress(gsl::not_null<tnsr::IJ<DataVector, Dim>*> stress,
              const tnsr::ii<DataVector, Dim>& strain,
              const tnsr::I<DataVector, Dim>& x) const noexcept;
  /// @}
};

}  // namespace ConstitutiveRelations
}  // namespace Elasticity

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/CubicCrystal.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"  // IWYU pragma: keep
