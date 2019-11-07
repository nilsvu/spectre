// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Protocols.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ProtocolTestHelpers.hpp"

namespace {

struct SomeField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct AnotherField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/// [evolution_analytic_sol_example]
template <size_t Dim>
struct EvolutionAnalyticSolution : evolution::protocols::AnalyticSolution {
 private:
  using PrecomputedData = double;

  // Functions computing individual variables
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, Dim>& x, const double t,
      const PrecomputedData& precomputed_data,
      tmpl::list<SomeField> /*meta*/) noexcept {
    return {Scalar<DataVector>{precomputed_data * t * get<0>(x)}};
  }

  tuples::TaggedTuple<AnotherField> variables(
      const tnsr::I<DataVector, Dim>& x, const double t,
      const PrecomputedData& precomputed_data,
      tmpl::list<AnotherField> /*meta*/) noexcept {
    return {Scalar<DataVector>{2. * precomputed_data * t * get<0>(x)}};
  }

 public:
  static constexpr size_t volume_dim = Dim;
  using supported_tags = tmpl::list<SomeField, AnotherField>;

  // Function returning a collection of variables to conform to the protocol
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, Dim>& x,
                                         const double t,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    // Precompute data to share between variable computations
    const PrecomputedData precomputed_data = 3.;
    // Compute each variable and collect in a TaggedTuple
    return {
        get<Tags>(variables(x, t, precomputed_data, tmpl::list<Tags>{}))...};
  }
};
/// [evolution_analytic_sol_example]
test_protocol_conformance<EvolutionAnalyticSolution<3>,
                          evolution::protocols::AnalyticSolution>;

/// [elliptic_analytic_sol_example]
template <size_t Dim>
struct EllipticAnalyticSolution : elliptic::protocols::AnalyticSolution {
 private:
  using PrecomputedData = double;

  // Functions computing individual variables
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, Dim>& x,
      const PrecomputedData& precomputed_data,
      tmpl::list<SomeField> /*meta*/) noexcept {
    return {Scalar<DataVector>{precomputed_data * get<0>(x)}};
  }

  tuples::TaggedTuple<AnotherField> variables(
      const tnsr::I<DataVector, Dim>& x,
      const PrecomputedData& precomputed_data,
      tmpl::list<AnotherField> /*meta*/) noexcept {
    return {Scalar<DataVector>{2. * precomputed_data * get<0>(x)}};
  }

 public:
  static constexpr size_t volume_dim = Dim;
  using supported_tags = tmpl::list<SomeField, AnotherField>;

  // Function returning a collection of variables to conform to the protocol
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, Dim>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    // Precompute data to share between variable computations
    const PrecomputedData precomputed_data = 3.;
    // Compute each variable and collect in a TaggedTuple
    return {get<Tags>(variables(x, precomputed_data, tmpl::list<Tags>{}))...};
  }
};
/// [elliptic_analytic_sol_example]
test_protocol_conformance<EllipticAnalyticSolution<3>,
                          elliptic::protocols::AnalyticSolution>;

struct NoVolumeDim {};
static_assert(
    not evolution::protocols::AnalyticSolution::template is_conforming_v<
        NoVolumeDim>,
    "Failed testing is_conforming_v");
static_assert(
    not elliptic::protocols::AnalyticSolution::template is_conforming_v<
        NoVolumeDim>,
    "Failed testing is_conforming_v");

struct NoSupportedTags {
  static constexpr size_t volume_dim = 3;
};
static_assert(
    not evolution::protocols::AnalyticSolution::template is_conforming_v<
        NoSupportedTags>,
    "Failed testing is_conforming_v");
static_assert(
    not elliptic::protocols::AnalyticSolution::template is_conforming_v<
        NoSupportedTags>,
    "Failed testing is_conforming_v");

struct NoVariables {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField>;
};
static_assert(
    not evolution::protocols::AnalyticSolution::template is_conforming_v<
        NoVariables>,
    "Failed testing is_conforming_v");
static_assert(
    not elliptic::protocols::AnalyticSolution::template is_conforming_v<
        NoVariables>,
    "Failed testing is_conforming_v");

struct MismatchingVolumeDimWithTime {
  static constexpr size_t volume_dim = 2;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, 3>& x, double t,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
};
static_assert(
    not evolution::protocols::AnalyticSolution::template is_conforming_v<
        MismatchingVolumeDimWithTime>,
    "Failed testing is_conforming_v");
struct MismatchingVolumeDimWithoutTime {
  static constexpr size_t volume_dim = 2;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
};
static_assert(
    not elliptic::protocols::AnalyticSolution::template is_conforming_v<
        MismatchingVolumeDimWithoutTime>,
    "Failed testing is_conforming_v");

struct NotAllSupportedTagsWithTime {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<SomeField> /*meta*/) const noexcept;
};
static_assert(
    not evolution::protocols::AnalyticSolution::template is_conforming_v<
        NotAllSupportedTagsWithTime>,
    "Failed testing is_conforming_v");
struct NotAllSupportedTagsWithoutTime {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField> /*meta*/) const noexcept;
};
static_assert(
    not elliptic::protocols::AnalyticSolution::template is_conforming_v<
        NotAllSupportedTagsWithoutTime>,
    "Failed testing is_conforming_v");

struct AllSingleTagsWithTime {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<SomeField> /*meta*/) const noexcept;
  tuples::TaggedTuple<AnotherField> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<AnotherField> /*meta*/) const noexcept;
};
static_assert(evolution::protocols::AnalyticSolution::template is_conforming_v<
                  AllSingleTagsWithTime>,
              "Failed testing is_conforming_v");
struct AllSingleTagsWithoutTime {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField> /*meta*/) const noexcept;
  tuples::TaggedTuple<AnotherField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<AnotherField> /*meta*/) const noexcept;
};
static_assert(elliptic::protocols::AnalyticSolution::template is_conforming_v<
                  AllSingleTagsWithoutTime>,
              "Failed testing is_conforming_v");

struct NotAllSingleTagsWithTime {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<SomeField> /*meta*/) const noexcept;
};
static_assert(
    not evolution::protocols::AnalyticSolution::template is_conforming_v<
        NotAllSingleTagsWithTime>,
    "Failed testing is_conforming_v");
struct NotAllSingleTagsWithoutTime {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField> /*meta*/) const noexcept;
};
static_assert(
    not elliptic::protocols::AnalyticSolution::template is_conforming_v<
        NotAllSingleTagsWithoutTime>,
    "Failed testing is_conforming_v");

}  // namespace
