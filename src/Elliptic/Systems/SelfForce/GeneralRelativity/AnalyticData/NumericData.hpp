// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AnalyticData {

struct Interpolator {
  std::vector<double> r;
  std::vector<double> theta;
  std::vector<double> flat_data;
  intrp::UniformMultiLinearSpanInterpolation<2, 20> interpolator;

  Interpolator() = default;
  Interpolator(std::vector<double>&& r_in, std::vector<double>&& theta_in,
               std::vector<double>&& flat_data_in)
      : r(std::move(r_in)),
        theta(std::move(theta_in)),
        flat_data(std::move(flat_data_in)),
        interpolator({gsl::make_span(r), gsl::make_span(theta)},
                     gsl::make_span(flat_data),
                     Index<2>{r.size(), theta.size()}) {}
};

struct Interpolator1D {
  std::vector<double> coord;   // r for Top/Bottom, theta for Left/Right
  std::vector<double> flat_data;
  intrp::UniformMultiLinearSpanInterpolation<1, 40> interpolator;

  Interpolator1D() = default;
  Interpolator1D(std::vector<double>&& coord_in,
                 std::vector<double>&& flat_data_in)
      : coord(std::move(coord_in)),
        flat_data(std::move(flat_data_in)),
        interpolator({gsl::make_span(coord)},
                     gsl::make_span(flat_data),
                     Index<1>{coord.size()}) {}
};

class NumericData : public elliptic::analytic_data::Background,
                    public elliptic::analytic_data::InitialGuess {
 public:
  struct Filename {
    static constexpr Options::String help =
        "Filename of the dat file containing the numeric data";
    using type = std::string;
  };
  struct BlackHoleMass {
    static constexpr Options::String help =
        "Kerr mass parameter 'M' of the black hole";
    using type = double;
  };
  struct BlackHoleSpin {
    static constexpr Options::String help =
        "Kerr dimensionless spin parameter 'chi' of the black hole";
    using type = double;
  };
  struct OrbitalRadius {
    static constexpr Options::String help =
        "Radius 'r_0' of the circular orbit";
    using type = double;
  };
  struct MModeNumber {
    static constexpr Options::String help =
        "Mode number 'm' of the m-mode decomposition";
    using type = int;
  };
  struct HyperboloidalSlicingTransitions {
    static constexpr Options::String help =
        "Transition points for the boost function. Four values: start and end "
        "of the first transition (from -1 to 0), then start and end of the "
        "second transition (from 0 to 1).";
    using type = std::array<double, 4>;
  };
  struct PenetratingHorizon {
    static constexpr Options::String help =
        "If 'False', use tortoise radial coordinate where the Kerr horizon is "
        "at negative infinity. If 'True', use Boyer-Lindquist radial "
        "coordinate where the Kerr horizon is at r_+.";
    using type = bool;
  };
  struct Pi_2_Rotation {
    static constexpr Options::String help =
        "If 'True', multiply h5 data by 2 pi * rotation factor "
        "to match with Barry's puncture convention.";
    using type = bool;
  };
  using options =
      tmpl::list<Filename, BlackHoleMass, BlackHoleSpin, OrbitalRadius,
                 MModeNumber, HyperboloidalSlicingTransitions,
                 PenetratingHorizon, Pi_2_Rotation>;
  static constexpr Options::String help =
      "Numeric data for the effective source and singular field";

  NumericData() = default;
  NumericData(const NumericData&) = default;
  NumericData& operator=(const NumericData&) = default;
  NumericData(NumericData&&) = default;
  NumericData& operator=(NumericData&&) = default;
  ~NumericData() override = default;

  NumericData(std::string filename, double black_hole_mass,
              double black_hole_spin, double orbital_radius, int m_mode_number,
              std::array<double, 4> hyperboloidal_slicing_transitions,
              bool penetrating_horizon, bool pi_2_rotation);

  explicit NumericData(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericData);

  tnsr::I<double, 2> puncture_position() const;
  const CircularOrbit& circular_orbit() const { return circular_orbit_; }

  using background_tags =
      tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>;
  using source_tags = tmpl::list<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>;

  // Background
  tuples::tagged_tuple_from_typelist<background_tags> variables(
      const tnsr::I<DataVector, 2>& x, background_tags /*meta*/) const;

  // Initial guess
  tuples::TaggedTuple<Tags::MMode> variables(
      const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/) const;

  // Fixed sources
  tuples::tagged_tuple_from_typelist<source_tags> variables(
      const tnsr::I<DataVector, 2>& x, source_tags /*meta*/) const {
    return variables(x, source_tags{}, true);
  }
  tuples::tagged_tuple_from_typelist<source_tags> variables(
      const tnsr::I<DataVector, 2>& x, source_tags /*meta*/,
      bool field_is_regularized) const;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 2>& x, const Mesh<2>& /*mesh*/,
      const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const NumericData& lhs, const NumericData& rhs);

  std::string filename_;
  std::array<Interpolator, 4> interpolators_;
  std::array<Interpolator1D, 4> boundary_interpolators_;
  CircularOrbit circular_orbit_;
  bool pi_2_rotation_{false};
};

bool operator!=(const NumericData& lhs, const NumericData& rhs);

}  // namespace GrSelfForce::AnalyticData
