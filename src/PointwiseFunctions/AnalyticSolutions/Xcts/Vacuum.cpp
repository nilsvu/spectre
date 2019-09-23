// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Vacuum.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts {
namespace Solutions {

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>> Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
Vacuum::variables(const tnsr::I<DataType, 3>& x,
                  tmpl::list<::Tags::FixedSource<
                      Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<::Tags::deriv<
    Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                      Frame::Inertial>>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<::Tags::deriv<
    Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                      Frame::Inertial>>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                      tmpl::size_t<3>, Frame::Inertial>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                      tmpl::size_t<3>, Frame::Inertial>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataType>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<gr::Tags::Shift<3, Frame::Inertial, DataType>>>
Vacuum::variables(const tnsr::I<DataType, 3>& x,
                  tmpl::list<::Tags::Initial<
                      gr::Tags::Shift<3, Frame::Inertial, DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<gr::Tags::Shift<3, Frame::Inertial, DataType>>>
Vacuum::variables(const tnsr::I<DataType, 3>& x,
                  tmpl::list<::Tags::FixedSource<
                      gr::Tags::Shift<3, Frame::Inertial, DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>> Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::StressTrace<DataType>> Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>
Vacuum::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

bool operator==(const Vacuum& /*lhs*/, const Vacuum& /*rhs*/) { return true; }
bool operator!=(const Vacuum& /*lhs*/, const Vacuum& /*rhs*/) { return false; }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE_VARS(_, data)                                              \
  template tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DTYPE(data)>>       \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::ConformalFactor<DTYPE(data)>> /*meta*/)           \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>               \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>           \
  Vacuum::variables(const tnsr::I<DTYPE(data), 3>& x,                          \
                    tmpl::list<::Tags::FixedSource<                            \
                        Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/)   \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>, tmpl::size_t<3>, \
                    Frame::Inertial>>                                          \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,       \
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)    \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,  \
                                    tmpl::size_t<3>, Frame::Inertial>>>        \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,              \
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::FixedSource<                            \
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>, tmpl::size_t<3>, \
                    Frame::Inertial>>>                                         \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,              \
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>                      \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>     \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>                         \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>> /*meta*/)           \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>>                        \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::FixedSource<                            \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>>                        \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>                        \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>> /*meta*/)   \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>>       \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>> /*meta*/)         \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>>   \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>> /*meta*/)         \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>>                \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>> /*meta*/)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>>>               \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::EnergyDensity<DTYPE(data)>>           \
  Vacuum::variables(const tnsr::I<DTYPE(data), 3>& x,                          \
                    tmpl::list<gr::Tags::EnergyDensity<DTYPE(data)>> /*meta*/) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::StressTrace<DTYPE(data)>>             \
  Vacuum::variables(const tnsr::I<DTYPE(data), 3>& x,                          \
                    tmpl::list<gr::Tags::StressTrace<DTYPE(data)>> /*meta*/)   \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>              \
  Vacuum::variables(                                                           \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,                 \
                                           DTYPE(data)>> /*meta*/)             \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VARS, (double, DataVector))

#undef DTYPE
#undef STYPE
#undef INSTANTIATE
#undef INSTANTIATE_VARS
}  // namespace Solutions
}  // namespace Xcts
/// \endcond
