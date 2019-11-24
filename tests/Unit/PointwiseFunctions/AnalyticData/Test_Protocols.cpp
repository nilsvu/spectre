// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticData/Protocols.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ProtocolTestHelpers.hpp"

namespace {

struct SomeField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct AnotherField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace

namespace test_evolution_protocols_analytic_data {

/// [evolution_analytic_data_example]
template <size_t Dim>
struct EvolutionAnalyticData : evolution::protocols::AnalyticData {
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
/// [evolution_analytic_data_example]

static_assert(test_protocol_conformance<EvolutionAnalyticData<3>,
                                        evolution::protocols::AnalyticData>,
              "Failed testing protocol conformance");

struct NoVolumeDim {};
static_assert(not evolution::protocols::AnalyticData::template is_conforming_v<
                  NoVolumeDim>,
              "Failed testing is_conforming_v");

struct NoSupportedTags {
  static constexpr size_t volume_dim = 3;
};
static_assert(not evolution::protocols::AnalyticData::template is_conforming_v<
                  NoSupportedTags>,
              "Failed testing is_conforming_v");

struct NoVariables {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField>;
};
static_assert(not evolution::protocols::AnalyticData::template is_conforming_v<
                  NoVariables>,
              "Failed testing is_conforming_v");

struct MismatchingVolumeDim {
  static constexpr size_t volume_dim = 2;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
};
static_assert(not evolution::protocols::AnalyticData::template is_conforming_v<
                  MismatchingVolumeDim>,
              "Failed testing is_conforming_v");

struct NotAllSupportedTags {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField> /*meta*/) const noexcept;
};
static_assert(not evolution::protocols::AnalyticData::template is_conforming_v<
                  NotAllSupportedTags>,
              "Failed testing is_conforming_v");

struct AllSingleTags {
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
static_assert(
    evolution::protocols::AnalyticData::template is_conforming_v<AllSingleTags>,
    "Failed testing is_conforming_v");

struct NotAllSingleTags {
  static constexpr size_t volume_dim = 3;
  using supported_tags = tmpl::list<SomeField, AnotherField>;
  tuples::TaggedTuple<SomeField, AnotherField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField, AnotherField> /*meta*/) const noexcept;
  tuples::TaggedTuple<SomeField> variables(
      const tnsr::I<DataVector, volume_dim>& x,
      tmpl::list<SomeField> /*meta*/) const noexcept;
};
static_assert(not evolution::protocols::AnalyticData::template is_conforming_v<
                  NotAllSingleTags>,
              "Failed testing is_conforming_v");

}  // namespace test_evolution_protocols_analytic_data

namespace test_elliptic_protocols_analytic_data {

/// [elliptic_analytic_data_example]
template <size_t Dim>
struct EllipticAnalyticData : elliptic::protocols::AnalyticData {
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
/// [elliptic_analytic_data_example]

static_assert(test_protocol_conformance<EllipticAnalyticData<3>,
                                        elliptic::protocols::AnalyticData>,
              "Failed testing protocol conformance");

struct NoVolumeDim {};
static_assert(not elliptic::protocols::AnalyticData::template is_conforming_v<
                  NoVolumeDim>,
              "Failed testing is_conforming_v");

struct NoVariables {
  static constexpr size_t volume_dim = 3;
};
static_assert(not elliptic::protocols::AnalyticData::template is_conforming_v<
                  NoVariables>,
              "Failed testing is_conforming_v");

}  // namespace test_elliptic_protocols_analytic_data
