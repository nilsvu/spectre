// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Punctures/MultiplePunctures.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"               // IWYU pragma: keep
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Options/Options.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace Punctures {
namespace InitialGuesses {

void MultiplePunctures::pup(PUP::er& p) noexcept { p | punctures; }

template <typename DataType>
tuples::TaggedTuple<::Punctures::Tags::Field<DataType>>
MultiplePunctures::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Punctures::Tags::Field<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<::Punctures::Tags::Field<DataType>>>
MultiplePunctures::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<::Punctures::Tags::Field<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<
    ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DataType>>>
MultiplePunctures::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<::Punctures::Tags::FieldGradient<
        3, Frame::Inertial, DataType>>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Source<::Punctures::Tags::Field<DataType>>>
MultiplePunctures::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Source<::Punctures::Tags::Field<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Source<
    ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DataType>>>
MultiplePunctures::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Source<::Punctures::Tags::FieldGradient<
        3, Frame::Inertial, DataType>>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Punctures::Tags::Alpha<DataType>>
MultiplePunctures::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Punctures::Tags::Alpha<DataType>> /*meta*/) const noexcept {
  auto one_over_alpha = make_with_value<Scalar<DataType>>(x, 0.);
  for (auto& puncture : punctures) {
    auto r = x;
    get<0>(r) -= puncture.center[0];
    get<1>(r) -= puncture.center[1];
    get<2>(r) -= puncture.center[2];
    get(one_over_alpha) += puncture.mass / get(magnitude(r));
  }
  return {Scalar<DataType>{1. / get(one_over_alpha)}};
}

template <typename DataType>
tuples::TaggedTuple<::Punctures::Tags::Beta<DataType>>
MultiplePunctures::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Punctures::Tags::Beta<DataType>> /*meta*/) const noexcept {
  auto A = make_with_value<tnsr::II<DataType, 3, Frame::Inertial>>(x, 0.);
  for (auto& puncture : punctures) {
    auto r = x;
    get<0>(r) -= puncture.center[0];
    get<1>(r) -= puncture.center[1];
    get<2>(r) -= puncture.center[2];
    const auto r_mag = get(magnitude(r));
    auto n = r;
    get<0>(n) /= r_mag;
    get<1>(n) /= r_mag;
    get<2>(n) /= r_mag;
    auto momentum_along_n = make_with_value<DataType>(x, 0.);
    for (size_t k = 0; k < 3; k++) {
      momentum_along_n += n.get(k) * puncture.momentum[k];
    }
    auto spin_around_n =
        make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.);
    for (LeviCivitaIterator<3> levi_civita_it; levi_civita_it;
         ++levi_civita_it) {
      const size_t j = levi_civita_it()[0];
      const size_t k = levi_civita_it()[1];
      const size_t l = levi_civita_it()[2];
      spin_around_n.get(j) =
          levi_civita_it.sign() * puncture.spin[k] * n.get(l);
    }
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        // Momentum
        A.get(i, j) +=
            3. / 2. / square(r_mag) *
            (puncture.momentum[i] * n.get(j) + puncture.momentum[j] * n.get(i) +
             n.get(i) * n.get(j) * momentum_along_n);
        // Spin
        A.get(i, j) +=
            3. / cube(r_mag) *
            (n.get(i) * spin_around_n.get(j) + n.get(j) * spin_around_n.get(i));
      }
      // Diagonal momentum
      A.get(i, i) -= 3. / 2. / square(r_mag) * momentum_along_n;
    }
  }
  auto A_squared = make_with_value<DataType>(x, 0.);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      A_squared += square(A.get(i, j));
    }
  }
  const auto alpha = get<::Punctures::Tags::Alpha<DataType>>(
      variables(x, tmpl::list<::Punctures::Tags::Alpha<DataType>>{}));
  return {Scalar<DataType>{pow<7>(get(alpha)) / 8. * A_squared}};
}

bool operator==(const MultiplePunctures::Puncture& lhs,
                const MultiplePunctures::Puncture& rhs) noexcept {
  return lhs.mass == rhs.mass and lhs.center == rhs.center and
         lhs.momentum == rhs.momentum and lhs.spin == rhs.spin;
}

bool operator!=(const MultiplePunctures::Puncture& lhs,
                const MultiplePunctures::Puncture& rhs) noexcept {
  return not(lhs == rhs);
}
bool operator==(const MultiplePunctures& lhs,
                const MultiplePunctures& rhs) noexcept {
  return lhs.punctures == rhs.punctures;
}

bool operator!=(const MultiplePunctures& lhs,
                const MultiplePunctures& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<::Punctures::Tags::Field<DTYPE(data)>>          \
  MultiplePunctures::variables(                                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Punctures::Tags::Field<DTYPE(data)>>) const noexcept;       \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<::Punctures::Tags::Field<DTYPE(data)>>>                  \
  MultiplePunctures::variables(                                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<::Punctures::Tags::Field<DTYPE(data)>>>)      \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DTYPE(data)>>>      \
  MultiplePunctures::variables(                                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<                                              \
          ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DTYPE(data)>>>) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::Source<::Punctures::Tags::Field<DTYPE(data)>>>                   \
  MultiplePunctures::variables(                                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Source<::Punctures::Tags::Field<DTYPE(data)>>>)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::Source<                                 \
      ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DTYPE(data)>>>      \
  MultiplePunctures::variables(                                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Source<                                               \
          ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DTYPE(data)>>>) \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Punctures::Tags::Alpha<DTYPE(data)>>          \
  MultiplePunctures::variables(                                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Punctures::Tags::Alpha<DTYPE(data)>>) const noexcept;       \
  template tuples::TaggedTuple<::Punctures::Tags::Beta<DTYPE(data)>>           \
  MultiplePunctures::variables(                                                \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Punctures::Tags::Beta<DTYPE(data)>>) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace InitialGuesses
}  // namespace Punctures
/// \endcond
