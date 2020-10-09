// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::Solutions {

enum class SchwarzschildCoordinates {
  /*!
   * \brief Isotropic Schwarzschild coordinates
   *
   * Eq. 1.60 in \cite BaumgarteShapiro
   *
   * Horizon at r = M / 2
   *
   * Maximally sliced (same as Schwarzschild), singularity-avoiding,
   */
  Isotropic,

  /*!
   * \brief Painlevé-Gullstrand coordinates
   */
  PainleveGullstrand,

  /*!
   * \brief Kerr-Schild (or ingoing Eddington-Finkelstein) coordinates with an
   * isotropic radial transformation
   *
   * See Haralds thesis, 7.4.1
   *
   * \f{align}
   * \hat{r} = \frac{r}{4} \left( 1+\sqrt{1+\frac{2}{r}} \right)^2 \exp\left(
   * 2 - 2\sqrt{1+\frac{2}{r}} \right)
   * \f}
   *
   * Conformally flat, but not maximally-sliced and with non-zero shift
   */
  KerrSchildIsotropic
};

std::ostream& operator<<(std::ostream& os,
                         const SchwarzschildCoordinates& coords) noexcept;

template <SchwarzschildCoordinates Coords>
class Schwarzschild
    : public tt::ConformsTo<elliptic::protocols::AnalyticSolution> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Schwarzschild spacetime in general relativity"};

  Schwarzschild() = default;
  Schwarzschild(const Schwarzschild&) noexcept = default;
  Schwarzschild& operator=(const Schwarzschild&) noexcept = default;
  Schwarzschild(Schwarzschild&&) noexcept = default;
  Schwarzschild& operator=(Schwarzschild&&) noexcept = default;
  ~Schwarzschild() noexcept = default;

  static double radius_at_horizon() noexcept;

  // @{
  /// Retrieve variable at coordinates `x`
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftBackground<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftStrain<3, Frame::Inertial,
                                                    DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<
                     Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> {
    return {make_with_value<Scalar<DataType>>(x, 1.)};
    // return {get<Xcts::Tags::ConformalFactor<DataType>>(
    //     variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{}))};
  }

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<
                     ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                        tmpl::size_t<3>, Frame::Inertial>>> {
    return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
    // return {get<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
    //                           tmpl::size_t<3>, Frame::Inertial>>(
    //     variables(
    //         x,
    //         tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
    //                                     tmpl::size_t<3>,
    //                                     Frame::Inertial>>{}))};
  }

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DataType>>> {
    return {make_with_value<Scalar<DataType>>(x, 1.)};
    // return {get<Xcts::Tags::LapseTimesConformalFactor<DataType>>(variables(
    //     x, tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>>{}))};
  }

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::Initial<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Tags::Initial<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>> {
    return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
    // return
    // {get<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
    //                           tmpl::size_t<3>, Frame::Inertial>>(
    //     variables(
    //         x,
    //         tmpl::list<
    //             ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
    //                           tmpl::size_t<3>, Frame::Inertial>>{}))};
  }

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Initial<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>> {
    return {make_with_value<tnsr::I<DataType, 3>>(x, 0.)};
  }

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<Xcts::Tags::ShiftStrain<
                     3, Frame::Inertial, DataType>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Initial<
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>> {
    return {make_with_value<tnsr::ii<DataType, 3>>(x, 0.)};
  }

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,
                                                      DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> boundary_variables(
      const tnsr::I<DataType, 3>& /*x*/, const Direction<3>& /*direction*/,
      const tnsr::i<DataType, 3>& /*face_normal*/,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    ASSERT(false, "Not implemented");
  }

  template <typename Tag>
  static elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, 3>& x, const Direction<3>& /*direction*/,
      Tag /*meta*/) {
    const DataVector r = get(magnitude(x));
    if (r[0] > 1.5 * radius_at_horizon()) {
      return elliptic::BoundaryCondition::Dirichlet;
    }
    if constexpr (std::is_same_v<Tag,
                                 Xcts::Tags::ConformalFactor<DataVector>>) {
      return elliptic::BoundaryCondition::Neumann;
    } else {
      return elliptic::BoundaryCondition::Dirichlet;
    }
  }

  template <typename Fields, typename Fluxes>
  void impose_boundary_conditions(
      const gsl::not_null<Variables<Fields>*> dirichlet_fields,
      const gsl::not_null<Variables<Fluxes>*> neumann_fields,
      const Variables<Fields>& vars, const Variables<Fluxes>& n_dot_fluxes,
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      const Direction<3>& /*direction*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& face_normal)
      const noexcept {
    const DataVector r = get(magnitude(x));
    if (r[0] > 1.5 * radius_at_horizon()) {
      get<Xcts::Tags::ConformalFactor<DataVector>>(*dirichlet_fields) =
          get<Xcts::Tags::ConformalFactor<DataVector>>(variables(
              x, tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>{}));
      get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(
          *dirichlet_fields) =
          get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(variables(
              x,
              tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataVector>>{}));
      if constexpr (tmpl::list_contains_v<
                        Fields, Xcts::Tags::ShiftExcess<DataVector, 3,
                                                        Frame::Inertial>>) {
        get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
            *dirichlet_fields) =
            get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
                variables(
                    x, tmpl::list<Xcts::Tags::ShiftExcess<DataVector, 3,
                                                          Frame::Inertial>>{}));
      }
    } else {
      const auto& conformal_factor =
          get(get<Xcts::Tags::ConformalFactor<DataVector>>(vars));
      const auto& lapse_times_conformal_factor =
          get(get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(vars));
      const DataVector K =
          get(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(variables(
              x, tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>>{})));

      // Conformal factor
      get(get<::Tags::NormalDotFlux<Xcts::Tags::ConformalFactor<DataVector>>>(
          *neumann_fields)) =
          conformal_factor / 2. / r - K * cube(conformal_factor) / 6.;
      if constexpr (tmpl::list_contains_v<
                        Fields, Xcts::Tags::ShiftExcess<DataVector, 3,
                                                        Frame::Inertial>>) {
        const auto& n_dot_longitudinal_shift = get<::Tags::NormalDotFlux<
            Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>>(
            n_dot_fluxes);
        Scalar<DataVector> nn_dot_longitudinal_shift{x.begin()->size()};
        normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                        n_dot_longitudinal_shift);
        get(get<::Tags::NormalDotFlux<Xcts::Tags::ConformalFactor<DataVector>>>(
            *neumann_fields)) += pow<4>(conformal_factor) / 8. /
                                 lapse_times_conformal_factor *
                                 get(nn_dot_longitudinal_shift);
      }

      // Lapse
      get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(
          *dirichlet_fields) =
          get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(variables(
              x,
              tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataVector>>{}));

      // Shift
      if constexpr (tmpl::list_contains_v<
                        Fields, Xcts::Tags::ShiftExcess<DataVector, 3,
                                                        Frame::Inertial>>) {
        DataVector beta_orthogonal =
            -lapse_times_conformal_factor / cube(conformal_factor);
        for (size_t i = 0; i < 3; ++i) {
          get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
              *dirichlet_fields)
              .get(i) = beta_orthogonal * face_normal.get(i);
        }
      }
    }
  }

  void pup(PUP::er& /* p */) noexcept {}  // NOLINT
};

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator==(
    const Schwarzschild<Coords>& /*lhs*/,
    const Schwarzschild<Coords>& /*rhs*/) noexcept {
  return true;
}

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const Schwarzschild<Coords>& /*lhs*/,
    const Schwarzschild<Coords>& /*rhs*/) noexcept {
  return false;
}

}  // namespace Xcts::Solutions
