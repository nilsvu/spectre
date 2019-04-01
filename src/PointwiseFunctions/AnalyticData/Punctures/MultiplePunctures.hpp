// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "Elliptic/Systems/Punctures/Tags.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Punctures {
namespace InitialGuesses {

class MultiplePunctures {
 public:
  struct Puncture {
    struct Mass {
      using type = double;
      static constexpr OptionString help{"The puncture mass"};
      static double lower_bound() noexcept { return 0.; }
    };
    struct Center {
      using type = std::array<double, 3>;
      static constexpr OptionString help{"The center of the puncture"};
    };
    struct Momentum {
      using type = std::array<double, 3>;
      static constexpr OptionString help{"The momentum of the puncture"};
    };
    struct Spin {
      using type = std::array<double, 3>;
      static constexpr OptionString help{"The spin of the puncture"};
    };
    using options = tmpl::list<Mass, Center, Momentum, Spin>;
    static constexpr OptionString help{"A puncture"};

    Puncture() = default;
    Puncture(const Puncture&) noexcept = delete;
    Puncture& operator=(const Puncture&) noexcept = delete;
    Puncture(Puncture&&) noexcept = default;
    Puncture& operator=(Puncture&&) noexcept = default;
    ~Puncture() noexcept = default;

    Puncture(Mass::type local_mass, Center::type local_center,
             Momentum::type local_momentum, Spin::type local_spin) noexcept
        : mass(std::move(local_mass)),
          center(std::move(local_center)),
          momentum(std::move(local_momentum)),
          spin(std::move(local_spin)) {}

    Mass::type mass = std::numeric_limits<double>::signaling_NaN();
    Center::type center{{std::numeric_limits<double>::signaling_NaN()}};
    Momentum::type momentum{{std::numeric_limits<double>::signaling_NaN()}};
    Spin::type spin{{std::numeric_limits<double>::signaling_NaN()}};

    void pup(PUP::er& p) noexcept {
      p | mass;
      p | center;
      p | momentum;
      p | spin;
    }
  };

  struct Punctures {
    using type = std::vector<Puncture>;
    static constexpr OptionString help{"The punctures"};
  };

  using options = tmpl::list<Punctures>;
  static constexpr OptionString help{"Punctures in GR"};

  MultiplePunctures() = default;
  MultiplePunctures(const MultiplePunctures&) noexcept = delete;
  MultiplePunctures& operator=(const MultiplePunctures&) noexcept = delete;
  MultiplePunctures(MultiplePunctures&&) noexcept = default;
  MultiplePunctures& operator=(MultiplePunctures&&) noexcept = default;
  ~MultiplePunctures() noexcept = default;

  MultiplePunctures(Punctures::type local_punctures) noexcept
      : punctures(std::move(local_punctures)) {}

  // @{
  /// Retrieve variable at coordinates `x`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Punctures::Tags::Field<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Punctures::Tags::Field<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::Initial<::Punctures::Tags::Field<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<::Punctures::Tags::Field<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<::Punctures::Tags::FieldGradient<
                     3, Frame::Inertial, DataType>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Initial<
          ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::Source<::Punctures::Tags::Field<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::Source<::Punctures::Tags::Field<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Source<::Punctures::Tags::FieldGradient<
                     3, Frame::Inertial, DataType>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Source<
          ::Punctures::Tags::FieldGradient<3, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Punctures::Tags::Alpha<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Punctures::Tags::Alpha<DataType>>;
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Punctures::Tags::Beta<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Punctures::Tags::Beta<DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

  Punctures::type punctures{};
};

bool operator==(const MultiplePunctures::Puncture& /*lhs*/,
                const MultiplePunctures::Puncture& /*rhs*/) noexcept;

bool operator!=(const MultiplePunctures::Puncture& lhs,
                const MultiplePunctures::Puncture& rhs) noexcept;

bool operator==(const MultiplePunctures& /*lhs*/,
                const MultiplePunctures& /*rhs*/) noexcept;

bool operator!=(const MultiplePunctures& lhs,
                const MultiplePunctures& rhs) noexcept;

}  // namespace InitialGuesses
}  // namespace Punctures
