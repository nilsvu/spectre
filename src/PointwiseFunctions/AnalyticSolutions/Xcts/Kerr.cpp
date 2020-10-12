// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace Xcts::Solutions {

std::ostream& operator<<(std::ostream& os,
                         const KerrCoordinates& coords) noexcept {
  switch (coords) {
    case KerrCoordinates::KerrSchild:
      return os << "KerrSchild";
    default:
      ERROR("Unknown KerrCoordinates");
  }
}

template <KerrCoordinates Coords>
Kerr<Coords>::Kerr(const double mass, std::array<double, 3> dimensionless_spin,
                   std::array<double, 3> center,
                   const Options::Context& context)
    : kerr_schild_solution_{mass, std::move(dimensionless_spin),
                            std::move(center), context} {}

// Conformal metric

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>
Kerr<Coords>::variables(const tnsr::I<DataType, 3>& x,
                        tmpl::list<Xcts::Tags::ConformalMetric<
                            DataType, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  const auto conformal_factor = get<Xcts::Tags::ConformalFactor<DataType>>(
      variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{}));
  auto conformal_metric = get<
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(
      kerr_schild_solution_.variables(
          x, 0.,
          tmpl::list<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>{}));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric.get(i, j) /= pow<4>(get(conformal_factor));
    }
  }
  return {std::move(conformal_metric)};
}

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>>
Kerr<Coords>::variables(const tnsr::I<DataType, 3>& x,
                        tmpl::list<Xcts::Tags::InverseConformalMetric<
                            DataType, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  const auto conformal_factor = get<Xcts::Tags::ConformalFactor<DataType>>(
      variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{}));
  auto inv_conformal_metric =
      get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>>(
          kerr_schild_solution_.variables(
              x, 0.,
              tmpl::list<gr::Tags::InverseSpatialMetric<3, Frame::Inertial,
                                                        DataType>>{}));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      inv_conformal_metric.get(i, j) *= pow<4>(get(conformal_factor));
    }
  }
  return {std::move(inv_conformal_metric)};
}

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>>
Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  const auto conformal_factor = get<Xcts::Tags::ConformalFactor<DataType>>(
      variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{}));
  const auto deriv_conformal_factor = get<::Tags::deriv<
      Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>(
      variables(x,
                tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                         tmpl::size_t<3>, Frame::Inertial>>{}));
  const auto conformal_metric = get<
      Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>(variables(
      x,
      tmpl::list<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>{}));
  const auto deriv_spatial_metric =
      get<::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>(
          kerr_schild_solution_.variables(
              x, 0.,
              tmpl::list<::Tags::deriv<
                  gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>{}));
  auto deriv_conformal_metric = deriv_spatial_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k <= j; ++k) {
        deriv_conformal_metric.get(i, j, k) /= pow<4>(get(conformal_factor));
        deriv_conformal_metric.get(i, j, k) -= 4. / get(conformal_factor) *
                                               conformal_metric.get(j, k) *
                                               deriv_conformal_factor.get(i);
      }
    }
  }
  return {std::move(deriv_conformal_metric)};
}

// Extrinsic curvature trace

template <>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
    noexcept {
  const auto vars = kerr_schild_solution_.variables(
      x, 0.,
      tmpl::list<
          gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>,
          gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>>{});
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>>(vars);
  const auto& inv_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>>(vars);
  return {trace(extrinsic_curvature, inv_spatial_metric)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

// Conformal factor

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.5)};
}

// Conformal factor gradient

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3>>(x, 0.)};
}

// Lapse (times conformal factor)

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/) const
    noexcept {
  const auto conformal_factor = get<Xcts::Tags::ConformalFactor<DataType>>(
      variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{}));
  auto lapse_times_conformal_factor =
      get<gr::Tags::Lapse<DataType>>(kerr_schild_solution_.variables(
          x, 0., tmpl::list<gr::Tags::Lapse<DataType>>{}));
  get(lapse_times_conformal_factor) *= get(conformal_factor);
  return {std::move(lapse_times_conformal_factor)};
}

// Lapse (times conformal factor) gradient

template <>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  const auto conformal_factor = get<Xcts::Tags::ConformalFactor<DataType>>(
      variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{}));
  const auto deriv_conformal_factor = get<::Tags::deriv<
      Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>(
      variables(x,
                tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                         tmpl::size_t<3>, Frame::Inertial>>{}));
  const auto vars = kerr_schild_solution_.variables(
      x, 0.,
      tmpl::list<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 gr::Tags::Lapse<DataType>>{});
  auto deriv_lapse_times_conformal_factor =
      get<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars);
  for (size_t i = 0; i < 3; ++i) {
    deriv_lapse_times_conformal_factor.get(i) *= get(conformal_factor);
    deriv_lapse_times_conformal_factor.get(i) +=
        get(get<gr::Tags::Lapse<DataType>>(vars)) *
        deriv_conformal_factor.get(i);
  }
  return {std::move(deriv_lapse_times_conformal_factor)};
}

// Shift background

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>
Kerr<Coords>::variables(const tnsr::I<DataType, 3>& x,
                        tmpl::list<Xcts::Tags::ShiftBackground<
                            DataType, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.)};
}

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>>
Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::II<DataType, 3>>(x, 0.)};
}

// Shift excess

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  return {get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(
      kerr_schild_solution_.variables(
          x, 0., tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>>{}))};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>
Kerr<KerrCoordinates::KerrSchild>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  const auto vars = kerr_schild_solution_.variables(
      x, 0.,
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>,
                 ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataType>,
                               tmpl::size_t<3>, Frame::Inertial>>{});
  const auto& shift = get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(vars);
  const auto& deriv_shift =
      get<::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>(vars);
  const auto conformal_metric = get<
      Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>(variables(
      x,
      tmpl::list<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>{}));
  const auto deriv_conformal_metric = get<
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>>(
      variables(x,
                tmpl::list<::Tags::deriv<
                    Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>>{}));
  const auto conformal_christoffel_first_kind =
      gr::christoffel_first_kind(deriv_conformal_metric);
  auto deriv_shift_lowered = make_with_value<tnsr::ij<DataType, 3>>(x, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_shift_lowered.get(i, j) +=
            conformal_metric.get(j, k) * deriv_shift.get(i, k) +
            shift.get(k) * deriv_conformal_metric.get(i, j, k);
      }
    }
  }
  auto shift_strain = make_with_value<tnsr::ii<DataType, 3>>(x, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      shift_strain.get(i, j) =
          0.5 * (deriv_shift_lowered.get(i, j) + deriv_shift_lowered.get(j, i));
      for (size_t k = 0; k < 3; ++k) {
        shift_strain.get(i, j) -=
            conformal_christoffel_first_kind.get(k, i, j) * shift.get(k);
      }
    }
  }
  return {std::move(shift_strain)};
}

// Fixed sources (all zero)

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>
Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.)};
}

// Matter sources (all zero)

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>> Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::StressTrace<DataType>> Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <KerrCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>
Kerr<Coords>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.)};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define COORDS(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ConformalMetric<DTYPE(data), 3, Frame::Inertial>>            \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<                                                              \
          Xcts::Tags::ConformalMetric<DTYPE(data), 3, Frame::Inertial>>)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::InverseConformalMetric<DTYPE(data), 3, Frame::Inertial>>     \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<Xcts::Tags::InverseConformalMetric<DTYPE(data), 3,            \
                                                    Frame::Inertial>>)         \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      Xcts::Tags::ConformalMetric<DTYPE(data), 3, Frame::Inertial>,            \
      tmpl::size_t<3>, Frame::Inertial>>                                       \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<::Tags::deriv<                                                \
          Xcts::Tags::ConformalMetric<DTYPE(data), 3, Frame::Inertial>,        \
          tmpl::size_t<3>, Frame::Inertial>>) const noexcept;                  \
  template tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>> \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>)              \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>>              \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>>)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DTYPE(data)>>       \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<Xcts::Tags::ConformalFactor<DTYPE(data)>>) const noexcept;    \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>, tmpl::size_t<3>, \
                    Frame::Inertial>>                                          \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,       \
                               tmpl::size_t<3>, Frame::Inertial>>)             \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>                      \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>                         \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<                                                              \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>>) const noexcept;    \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<           \
          DTYPE(data), 3, Frame::Inertial>>                                    \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<                                                              \
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<       \
              DTYPE(data), 3, Frame::Inertial>>) const noexcept;               \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftBackground<DTYPE(data), 3, Frame::Inertial>>            \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<                                                              \
          Xcts::Tags::ShiftBackground<DTYPE(data), 3, Frame::Inertial>>)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>                \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>)    \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftStrain<DTYPE(data), 3, Frame::Inertial>>                \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<Xcts::Tags::ShiftStrain<DTYPE(data), 3, Frame::Inertial>>)    \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>           \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<                                                              \
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>)      \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>)                \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::FixedSource<                            \
      Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>               \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::EnergyDensity<DTYPE(data)>>           \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<gr::Tags::EnergyDensity<DTYPE(data)>>) const noexcept;        \
  template tuples::TaggedTuple<gr::Tags::StressTrace<DTYPE(data)>>             \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<gr::Tags::StressTrace<DTYPE(data)>>) const noexcept;          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>              \
  Kerr<COORDS(data)>::variables(                                               \
      const tnsr::I<DTYPE(data), 3>&,                                          \
      tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>)  \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (KerrCoordinates::KerrSchild))
template class Kerr<KerrCoordinates::KerrSchild>;

#undef DTYPE
#undef INSTANTIATE

}  // namespace Xcts::Solutions
/// \endcond
