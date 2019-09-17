// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace Xcts {
namespace Solutions {

class Vacuum {
 public:
  using options = tmpl::list<>;

  static constexpr OptionString help = {"Just flat spacetime."};

  Vacuum() = default;
  Vacuum(const Vacuum& /*rhs*/) = delete;
  Vacuum& operator=(const Vacuum& /*rhs*/) = delete;
  Vacuum(Vacuum&& /*rhs*/) noexcept = default;
  Vacuum& operator=(Vacuum&& /*rhs*/) noexcept = default;
  ~Vacuum() = default;

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    return {get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  //  NOLINT

 private:
  template <typename DataType>
  using ConformalFactorGradient =
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>;
  template <typename DataType>
  using LapseTimesConformalFactorGradient =
      Xcts::Tags::LapseTimesConformalFactorGradient<3, Frame::Inertial,
                                                    DataType>;
  template <typename DataType>
  using Shift = gr::Tags::Shift<3, Frame::Inertial, DataType>;
  template <typename DataType>
  using ShiftStrain = Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>;
  template <typename DataType>
  using MomentumDensity =
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>;

 public:
#define FUNC_DECL(r, data, elem)                                     \
  template <typename DataType>                                       \
  tuples::TaggedTuple<elem> variables(const tnsr::I<DataType, 3>& x, \
                                      tmpl::list<elem> /*meta*/)     \
      const noexcept;                                                \
  template <typename DataType>                                       \
  tuples::TaggedTuple<::Tags::Initial<elem>> variables(              \
      const tnsr::I<DataType, 3>& x,                                 \
      tmpl::list<::Tags::Initial<elem>> /*meta*/) const noexcept;    \
  template <typename DataType>                                       \
  tuples::TaggedTuple<::Tags::FixedSource<elem>> variables(          \
      const tnsr::I<DataType, 3>& x,                                 \
      tmpl::list<::Tags::FixedSource<elem>> /*meta*/) const noexcept;

#define MY_LIST                                                         \
  BOOST_PP_TUPLE_TO_LIST(                                               \
      9, (Xcts::Tags::ConformalFactor<DataType>,                        \
          ConformalFactorGradient<DataType>,                            \
          Xcts::Tags::LapseTimesConformalFactor<DataType>,              \
          LapseTimesConformalFactorGradient<DataType>, Shift<DataType>, \
          ShiftStrain<DataType>, gr::Tags::EnergyDensity<DataType>,     \
          gr::Tags::StressTrace<DataType>, MomentumDensity<DataType>))

  BOOST_PP_LIST_FOR_EACH(FUNC_DECL, _, MY_LIST)
#undef MY_LIST
#undef FUNC_DECL
};

bool operator==(const Vacuum& /*lhs*/, const Vacuum& /*rhs*/);
bool operator!=(const Vacuum& /*lhs*/, const Vacuum& /*rhs*/);

}  // namespace Solutions
}  // namespace Xcts
