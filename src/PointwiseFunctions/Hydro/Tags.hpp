// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to hydrodynamic systems.
namespace hydro {

/// %Tags for options of hydrodynamic systems.
namespace OptionTags {

/// The equation of state of the fluid.
template <bool IsRelativistic, size_t ThermoDim>
struct EquationOfState {
  using type = std::unique_ptr<
      EquationsOfState::EquationOfState<IsRelativistic, ThermoDim>>;
  static constexpr Options::String help = {"The equation of state to use"};
};
}  // namespace OptionTags

/// %Tags for hydrodynamic systems.
namespace Tags {

/// The Alfvén speed squared \f$v_A^2\f$.
template <typename DataType>
struct AlfvenSpeedSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The magnetic field \f$b^\mu = u_\nu {}^\star\!F^{\mu \nu}\f$
/// measured by an observer comoving with the fluid with 4-velocity
/// \f$u_\nu\f$ where \f${}^\star\!F^{\mu \nu}\f$
/// is the dual of the Faraday tensor.  Note that \f$b^\mu\f$ has a
/// time component (that is, \f$b^\mu n_\mu \neq 0\f$, where \f$n_\mu\f$ is
/// the normal to the spacelike hypersurface).
template <typename DataType, size_t Dim, typename Fr>
struct ComovingMagneticField : db::SimpleTag {
  using type = tnsr::A<DataType, Dim, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "ComovingMagneticField";
  }
};

/// The square of the comoving magnetic field, \f$b^\mu b_\mu\f$
template <typename DataType>
struct ComovingMagneticFieldSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The magnitude of the comoving magnetic field, \f$\sqrt{b^\mu b_\mu}\f$
template <typename DataType>
struct ComovingMagneticFieldMagnitude : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The divergence-cleaning field \f$\Phi\f$.
template <typename DataType>
struct DivergenceCleaningField : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The electron fraction \f$Y_e\f$.
template <typename DataType>
struct ElectronFraction : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Base tag for the equation of state
struct EquationOfStateBase : db::BaseTag {};

/// The equation of state retrieved from the analytic solution / data in the
/// input file
template <typename EquationOfStateType>
struct EquationOfState : EquationOfStateBase, db::SimpleTag {
  using type = EquationOfStateType;

  template <typename Metavariables>
  using option_tags = tmpl::list<
      tmpl::front<typename Metavariables::initial_data_tag::option_tags>>;
  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  static type create_from_options(
      const typename Metavariables::initial_data_tag::type& initial_data) {
    return initial_data.equation_of_state().get_clone();
  }
};

/// The equation of state constructed from options in the input file
template <bool IsRelativistic, size_t ThermodynamicDim>
struct EquationOfStateFromOptions : EquationOfStateBase, db::SimpleTag {
  using type = std::unique_ptr<
      EquationsOfState::EquationOfState<IsRelativistic, ThermodynamicDim>>;

  static constexpr bool pass_metavariables = false;
  using option_tags =
      tmpl::list<OptionTags::EquationOfState<IsRelativistic, ThermodynamicDim>>;

  static type create_from_options(const type& eos) { return eos->get_clone(); }
};

/// The Lorentz factor \f$W = (1-v^iv_i)^{-1/2}\f$, where \f$v^i\f$ is
/// the spatial velocity of the fluid.
template <typename DataType>
struct LorentzFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The square of the Lorentz factor \f$W^2\f$.
template <typename DataType>
struct LorentzFactorSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The magnetic field \f$B^i = n_\mu {}^\star\!F^{i \mu}\f$ measured by an
/// Eulerian observer, where \f$n_\mu\f$ is the normal to the spatial
/// hypersurface and \f${}^\star\!F^{\mu \nu}\f$ is the dual of the
/// Faraday tensor.  Note that \f$B^i\f$ is purely spatial, and it
/// can be lowered using the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct MagneticField : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "MagneticField"; }
};

/// The magnetic field dotted into the spatial velocity, \f$B^iv_i\f$ where
/// \f$v_i\f$ is the spatial velocity one-form.
template <typename DataType>
struct MagneticFieldDotSpatialVelocity : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The one-form of the magnetic field.  Note that \f$B^i\f$ is raised
/// and lowered with the spatial metric.
/// \see hydro::Tags::MagneticField
template <typename DataType, size_t Dim, typename Fr>
struct MagneticFieldOneForm : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "MagneticFieldOneForm";
  }
};

/// The square of the magnetic field, \f$B^iB_i\f$
template <typename DataType>
struct MagneticFieldSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The magnetic pressure \f$p_m\f$.
template <typename DataType>
struct MagneticPressure : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The fluid pressure \f$p\f$.
template <typename DataType>
struct Pressure : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The rest-mass density \f$\rho\f$.
template <typename DataType>
struct RestMassDensity : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The sound speed squared \f$c_s^2\f$.
template <typename DataType>
struct SoundSpeedSquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The spatial velocity \f$v^i\f$ of the fluid,
/// where \f$v^i=u^i/W + \beta^i/\alpha\f$.
/// Here \f$u^i\f$ is the spatial part of the 4-velocity of the fluid,
/// \f$W\f$ is the Lorentz factor, \f$\beta^i\f$ is the shift vector,
/// and \f$\alpha\f$ is the lapse function. Note that \f$v^i\f$ is raised
/// and lowered with the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct SpatialVelocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "SpatialVelocity"; }
};

/// The spatial velocity one-form \f$v_i\f$, where \f$v_i\f$ is raised
/// and lowered with the spatial metric.
template <typename DataType, size_t Dim, typename Fr>
struct SpatialVelocityOneForm : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Fr>;
  static std::string name() {
    return Frame::prefix<Fr>() + "SpatialVelocityOneForm";
  }
};

/// The spatial velocity squared \f$v^2 = v_i v^i\f$.
template <typename DataType>
struct SpatialVelocitySquared : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The relativistic specific enthalpy \f$h\f$.
template <typename DataType>
struct SpecificEnthalpy : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The specific internal energy \f$\epsilon\f$.
template <typename DataType>
struct SpecificInternalEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// The temperature \f$T\f$ of the fluid.
template <typename DataType>
struct Temperature : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// The spatial components of the four-velocity one-form \f$u_i\f$.
template <typename DataType, size_t Dim, typename Fr>
struct LowerSpatialFourVelocity : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Fr>;
};

/// The Lorentz factor \f$W\f$ times the spatial velocity \f$v^i\f$.
template <typename DataType, size_t Dim, typename Fr>
struct LorentzFactorTimesSpatialVelocity : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
};

/// The vector \f$J^i\f$ in \f$\dot{M} = -\int J^i s_i d^2S\f$,
/// representing the mass flux through a surface with normal \f$s_i\f$.
///
/// Note that the integral is understood
/// as a flat-space integral: all metric factors are included in \f$J^i\f$.
/// In particular, if the integral is done over a Strahlkorper, the
/// `StrahlkorperGr::euclidean_area_element` of the Strahlkorper should be used,
/// and \f$s_i\f$ is
/// the normal one-form to the Strahlkorper normalized with the flat metric,
/// \f$s_is_j\delta^{ij}=1\f$.
///
/// The formula is
/// \f$ J^i = \rho W \sqrt{\gamma}(\alpha v^i-\beta^i)\f$,
/// where \f$\rho\f$ is the mass density, \f$W\f$ is the Lorentz factor,
/// \f$v^i\f$ is the spatial velocity of the fluid,
/// \f$\gamma\f$ is the determinant of the 3-metric \f$\gamma_{ij}\f$,
/// \f$\alpha\f$ is the lapse, and \f$\beta^i\f$ is the shift.
template <typename DataType, size_t Dim, typename Fr>
struct MassFlux : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() { return Frame::prefix<Fr>() + "MassFlux"; }
};
}  // namespace Tags

}  // namespace hydro
