// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {

namespace detail {
struct BinaryVariables {
  using DataType = DataVector;
  static constexpr size_t Dim = 3;
  using Cache = CachedTempBuffer<
      BinaryVariables,
      Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>,
      Tags::ConformalChristoffelFirstKind<DataType, Dim, Frame::Inertial>,
      Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>,
      Tags::ConformalChristoffelContracted<DataType, Dim, Frame::Inertial>,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, Dim,
                                                              Frame::Inertial>>;

  const tnsr::I<DataVector, Dim>& x;
  const double angular_velocity;
  const tnsr::ii<DataVector, Dim>& conformal_metric;
  const tnsr::ijj<DataVector, Dim>& deriv_conformal_metric;

  void operator()(
      gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> conformal_christoffel_first_kind,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelFirstKind<
          DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::Ijj<DataType, Dim>*>
                      conformal_christoffel_second_kind,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalChristoffelSecondKind<
                      DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> conformal_christoffel_contracted,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelContracted<
          DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataVector, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::iJ<DataVector, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::II<DataVector, Dim>*> longitudinal_shift_background,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, Dim, Frame::Inertial> /*meta*/) const noexcept;
};
}  // namespace detail

template <typename IsolatedObjectRegistrars, typename Registrars>
class Binary;

namespace Registrars {
template <typename IsolatedObjectRegistrars>
struct Binary {
  template <typename Registrars>
  using f = Xcts::AnalyticData::Binary<IsolatedObjectRegistrars, Registrars>;
};
}  // namespace Registrars

template <typename IsolatedObjectRegistrars,
          typename Registrars = tmpl::list<
              Xcts::AnalyticData::Registrars::Binary<IsolatedObjectRegistrars>>>
class Binary : public AnalyticData<Registrars> {
 private:
  using Base = AnalyticData<Registrars>;

 public:
  using IsolatedObjectBase =
      Xcts::Solutions::AnalyticSolution<IsolatedObjectRegistrars>;

  struct XCoords {
    static constexpr Options::String help =
        "The coordinates on the x-axis where the two objects are places";
    using type = std::array<double, 2>;
  };

  struct ObjectA {
    static constexpr Options::String help =
        "The object placed along the negative x-axis";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };

  struct ObjectB {
    static constexpr Options::String help =
        "The object placed along the positive x-axis";
    using type = std::unique_ptr<IsolatedObjectBase>;
  };

  struct AngularVelocity {
    static constexpr Options::String help = "Orbital angular velocity";
    using type = double;
  };

  struct FalloffWidths {
    static constexpr Options::String help =
        "The widths for the window functions around the two objects";
    using type = std::array<double, 2>;
  };

  using options =
      tmpl::list<XCoords, ObjectA, ObjectB, AngularVelocity, FalloffWidths>;
  static constexpr Options::String help =
      "Binary compact-object initial data in general relativity";

  Binary() = default;
  Binary(const Binary&) noexcept = default;
  Binary& operator=(const Binary&) noexcept = default;
  Binary(Binary&&) noexcept = default;
  Binary& operator=(Binary&&) noexcept = default;
  ~Binary() noexcept = default;

  Binary(std::array<double, 2> xcoords,
         std::unique_ptr<IsolatedObjectBase> object_a,
         std::unique_ptr<IsolatedObjectBase> object_b, double angular_velocity,
         std::array<double, 2> falloff_widths) noexcept
      : xcoords_(xcoords),
        superposed_objects_({std::move(object_a), std::move(object_b)}),
        angular_velocity_(angular_velocity),
        falloff_widths_(falloff_widths) {}

  explicit Binary(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Binary);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    // These tags are computed from superpositions of the two isolated solutions
    using superposed_tags = tmpl::list<
        // Free data
        Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
        ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>,
        gr::Tags::TraceExtrinsicCurvature<DataType>,
        ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
        // Data for initial guess
        Tags::ConformalFactor<DataType>,
        Tags::LapseTimesConformalFactor<DataType>,
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
        // Matter sources
        gr::Tags::EnergyDensity<DataType>, gr::Tags::StressTrace<DataType>,
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
        ::Tags::FixedSource<Tags::ConformalFactor<DataType>>,
        ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>>,
        ::Tags::FixedSource<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;
    using non_superposed_tags = tmpl::list<
        Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>,
        Tags::ConformalChristoffelFirstKind<DataType, 3, Frame::Inertial>,
        Tags::ConformalChristoffelSecondKind<DataType, 3, Frame::Inertial>,
        Tags::ConformalChristoffelContracted<DataType, 3, Frame::Inertial>,
        Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
        Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataType, 3, Frame::Inertial>>;
    using requested_superposed_tags =
        tmpl::list_difference<tmpl::list<RequestedTags...>,
                              non_superposed_tags>;
    using requested_non_superposed_tags =
        tmpl::list_difference<tmpl::list<RequestedTags...>,
                              requested_superposed_tags>;
    static_assert(
        std::is_same_v<
            tmpl::list_difference<requested_superposed_tags, superposed_tags>,
            tmpl::list<>>,
        "Not all requested tags are supported.");
    using non_superposed_dependencies = tmpl::conditional_t<
        std::is_same_v<requested_non_superposed_tags, tmpl::list<>>,
        tmpl::list<>,
        tmpl::list<
            Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
            Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
            ::Tags::deriv<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                          tmpl::size_t<3>, Frame::Inertial>>>;
    using requested_isolated_tags = tmpl::remove_duplicates<
        tmpl::append<requested_superposed_tags, non_superposed_dependencies>>;
    std::array<tnsr::I<DataVector, 3>, 2> x_isolated{{x, x}};
    get<0>(x_isolated[0]) -= xcoords_[0];
    get<0>(x_isolated[1]) -= xcoords_[1];
    std::array<DataVector, 2> euclidean_distance{};
    euclidean_distance[0] = get(magnitude(x_isolated[0]));
    euclidean_distance[1] = get(magnitude(x_isolated[1]));
    const auto left_vars = superposed_objects_[0]->variables(
        x_isolated[0], requested_isolated_tags{});
    const auto right_vars = superposed_objects_[1]->variables(
        x_isolated[1], requested_isolated_tags{});
    const auto flat_vars = flatness_.variables(x, requested_isolated_tags{});
    tuples::tagged_tuple_from_typelist<requested_isolated_tags>
        superposed_vars{};
    // Compute superpositions
    const DataVector window_left =
        exp(-square(euclidean_distance[0]) / square(falloff_widths_[0]));
    const DataVector window_right =
        exp(-square(euclidean_distance[1]) / square(falloff_widths_[1]));
    tmpl::for_each<requested_isolated_tags>([&superposed_vars, &left_vars,
                                             &right_vars, &flat_vars,
                                             &window_left, &window_right](
                                                auto tag_v) noexcept {
      using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
      for (size_t i = 0; i < get<tag>(superposed_vars).size(); ++i) {
        get<tag>(superposed_vars)[i] =
            get<tag>(flat_vars)[i] +
            window_left * (get<tag>(left_vars)[i] - get<tag>(flat_vars)[i]) +
            window_right * (get<tag>(right_vars)[i] - get<tag>(flat_vars)[i]);
      }
    });
    if constexpr (tmpl::list_contains_v<
                      requested_superposed_tags,
                      ::Tags::deriv<
                          Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                          tmpl::size_t<3>, Frame::Inertial>>) {
      // Add derivative of window function
      const auto& conformal_metric_left =
          get<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(left_vars);
      const auto& conformal_metric_right =
          get<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
              right_vars);
      const auto& flat_metric =
          get<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(flat_vars);
      auto& deriv_conformal_metric = get<
          ::Tags::deriv<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                        tmpl::size_t<3>, Frame::Inertial>>(superposed_vars);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          for (size_t k = 0; k <= j; ++k) {
            deriv_conformal_metric.get(i, j, k) -=
                2. * x_isolated[0].get(i) / square(falloff_widths_[0]) *
                    window_left *
                    (conformal_metric_left.get(j, k) - flat_metric.get(j, k)) -
                2. * x_isolated[1].get(i) / square(falloff_widths_[1]) *
                    window_right *
                    (conformal_metric_right.get(j, k) - flat_metric.get(j, k));
          }
        }
      }
    }
    if constexpr (std::is_same_v<requested_non_superposed_tags, tmpl::list<>>) {
      return superposed_vars;
    } else {
      tuples::TaggedTuple<RequestedTags...> binary_vars{};
      // Fill in non-superposed tags
      typename detail::BinaryVariables::Cache non_superposed_cache{
          x.begin()->size(),
          detail::BinaryVariables{
              x, angular_velocity_,
              get<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
                  superposed_vars),
              get<::Tags::deriv<
                  Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>>(superposed_vars)}};
      tmpl::for_each<requested_non_superposed_tags>(
          [&non_superposed_cache, &binary_vars](auto tag_v) noexcept {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            get<tag>(binary_vars) = non_superposed_cache.get_var(tag{});
          });
      // Move remaining requested tags
      tmpl::for_each<requested_superposed_tags>(
          [&binary_vars, &superposed_vars](auto tag_v) noexcept {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            get<tag>(binary_vars) = std::move(get<tag>(superposed_vars));
          });
      return binary_vars;
    }
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    p | xcoords_;
    p | superposed_objects_;
    p | angular_velocity_;
    p | falloff_widths_;
  }

  double angular_velocity() const noexcept { return angular_velocity_; }

 private:
  std::array<double, 2> xcoords_;
  std::array<std::unique_ptr<IsolatedObjectBase>, 2> superposed_objects_;
  Xcts::Solutions::Flatness<> flatness_{};
  double angular_velocity_;
  std::array<double, 2> falloff_widths_;
};

/// \cond
template <typename IsolatedObjectRegistrars, typename Registrars>
PUP::able::PUP_ID Binary<IsolatedObjectRegistrars, Registrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Xcts::AnalyticData
