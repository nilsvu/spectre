// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Punctures {
namespace AnalyticData {

class MultiplePunctures : public MarkAsAnalyticData {
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
  auto variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                 tmpl::list<Tags::Field> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Field>;

  auto variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<Tags::Field>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Tags::Initial<Tags::Field>>;

  auto variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<::Tags::deriv<
                     Tags::Field, tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<::Tags::Initial<
          ::Tags::deriv<Tags::Field, tmpl::size_t<3>, Frame::Inertial>>>;

  auto variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<Tags::Field>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Tags::FixedSource<Tags::Field>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  Variables<tmpl::list<Tags::Alpha, Tags::Beta>> alpha_and_beta(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept;

  struct BackgroundFieldsCompute
      : ::Tags::Variables<tmpl::list<Tags::Alpha, Tags::Beta>>,
        db::ComputeTag {
    using base = ::Tags::Variables<tmpl::list<Tags::Alpha, Tags::Beta>>;
    using argument_tags =
        tmpl::list<::Tags::AnalyticSolution<MultiplePunctures>,
                   ::Tags::Coordinates<3, Frame::Inertial>>;
    static db::item_type<base> function(
        const MultiplePunctures& solution,
        const tnsr::I<DataVector, 3, Frame::Inertial>&
            inertial_coords) noexcept {
      return solution.alpha_and_beta(inertial_coords);
    }
  };

  using compute_tags = tmpl::list<BackgroundFieldsCompute>;

  using observe_fields = tmpl::list<Tags::Alpha, Tags::Beta>;

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

}  // namespace AnalyticData
}  // namespace Punctures
