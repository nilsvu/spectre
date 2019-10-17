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
#include "Options/Options.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Punctures {
namespace AnalyticData {

void MultiplePunctures::pup(PUP::er& p) noexcept { p | punctures; }

tuples::TaggedTuple<Tags::Field> MultiplePunctures::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<Tags::Field> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataVector>>(x, 0.)};
}

tuples::TaggedTuple<::Tags::Initial<Tags::Field>> MultiplePunctures::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<Tags::Field>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataVector>>(x, 0.)};
}

tuples::TaggedTuple<::Tags::Initial<
    ::Tags::deriv<Tags::Field, tmpl::size_t<3>, Frame::Inertial>>>
MultiplePunctures::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<::Tags::deriv<Tags::Field, tmpl::size_t<3>,
                                             Frame::Inertial>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(x, 0.)};
}

tuples::TaggedTuple<::Tags::FixedSource<Tags::Field>>
MultiplePunctures::variables(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::FixedSource<Tags::Field>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataVector>>(x, 0.)};
}

Variables<tmpl::list<Tags::Alpha, Tags::Beta>>
MultiplePunctures::alpha_and_beta(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
  auto one_over_alpha = make_with_value<Scalar<DataVector>>(x, 0.);
  for (auto& puncture : punctures) {
    auto r = x;
    get<0>(r) -= puncture.center[0];
    get<1>(r) -= puncture.center[1];
    get<2>(r) -= puncture.center[2];
    get(one_over_alpha) += puncture.mass / get(magnitude(r));
  }
  Scalar<DataVector> alpha{1. / get(one_over_alpha)};

  auto A = make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(x, 0.);
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
    auto momentum_along_n = make_with_value<DataVector>(x, 0.);
    for (size_t k = 0; k < 3; k++) {
      momentum_along_n += n.get(k) * puncture.momentum[k];
    }
    auto spin_around_n =
        make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(x, 0.);
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
  auto A_squared = make_with_value<DataVector>(x, 0.);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      A_squared += square(A.get(i, j));
    }
  }
  Scalar<DataVector> beta{pow<7>(get(alpha)) / 8. * A_squared};

  auto result =
      make_with_value<Variables<tmpl::list<Tags::Alpha, Tags::Beta>>>(x, 0.);
  get<Tags::Alpha>(result) = std::move(alpha);
  get<Tags::Beta>(result) = std::move(beta);
  return result;
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

}  // namespace AnalyticData
}  // namespace Punctures
/// \endcond
