// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
struct DataVector;
/// \endcond

// Analytic data classes for elliptic and evolution systems have the same
// interface requirements, so they can share the code that checks protocol
// conformance. However, they are conceptually different: Evolution analytic
// data represents initial data that typically satisfies the constraints of the
// system, whereas elliptic analytic data represents only an initial guess for
// the elliptic solve. We define a protocol for each so we can distinguish
// between them. The distinction is more obvious for the analytic _solution_
// protocols that require a slightly different interface.
namespace analytic_data_protocols_detail {

CREATE_HAS_STATIC_MEMBER_VARIABLE(volume_dim)
CREATE_HAS_TYPE_ALIAS(supported_tags)
CREATE_IS_CALLABLE(variables)

template <typename ConformingType, typename TagsList>
struct IsVariablesCallable
    : is_variables_callable_r_t<
          tuples::tagged_tuple_from_typelist<TagsList>, ConformingType,
          tnsr::I<DataVector, ConformingType::volume_dim>, TagsList> {};
// All tags
template <typename ConformingType>
struct IsVariablesCallableWithAllTags
    : IsVariablesCallable<ConformingType,
                          typename ConformingType::supported_tags> {};
// Any single tag
template <typename ConformingType, typename TagsList>
struct IsVariablesCallableWithAnySingleTagImpl;
template <typename ConformingType, typename... Tags>
struct IsVariablesCallableWithAnySingleTagImpl<ConformingType,
                                               tmpl::list<Tags...>>
    : cpp17::conjunction<
          IsVariablesCallable<ConformingType, tmpl::list<Tags>>...> {};
template <typename ConformingType>
struct IsVariablesCallableWithAnySingleTag
    : IsVariablesCallableWithAnySingleTagImpl<
          ConformingType, typename ConformingType::supported_tags> {};

template <typename ConformingType>
static constexpr bool is_conforming_v = std::conditional_t<
    has_volume_dim_v<ConformingType, size_t> and
        has_supported_tags_v<ConformingType>,
    cpp17::conjunction<IsVariablesCallableWithAllTags<ConformingType>,
                       IsVariablesCallableWithAnySingleTag<ConformingType>>,
    std::false_type>::value;

}  // namespace analytic_data_protocols_detail

namespace evolution {
namespace protocols {

/*!
 * \ingroup ProtocolsGroup
 * \brief Analytic initial data for a system of evolution equations.
 *
 * Requires the class has these static member variables:
 * - `size_t volume_dim`: The spatial dimension of tensors for this solution
 *
 * Require the class has these type aliases:
 * - `supported_tags`: A typelist of tags that can be retrieved from the
 * `variables` function.
 *
 * Requires the class has these member functions:
 * - `variables`: Returns a collection of variables at coordinates `x`. The
 * particular set of variables to be computed is specified as a `tmpl::list` of
 * tags that is supplied as the third `meta` argument. Must support any subset
 * and permutation of the `supported_tags`. When multiple variables are
 * requested, this function should make sure that precomputed data is shared
 * among the computation of these variables so it can be evaluated efficiently.
 *
 * This is an example implementation of a class that conforms to this protocol:
 *
 * \snippet AnalyticData/Test_Protocols.cpp evolution_analytic_data_example
 */
struct AnalyticData {
  template <typename ConformingType>
  static constexpr bool is_conforming_v =
      analytic_data_protocols_detail::is_conforming_v<ConformingType>;
};

}  // namespace protocols
}  // namespace evolution

namespace elliptic {
namespace protocols {

/*!
 * \ingroup ProtocolsGroup
 * \brief An analytic initial guess to a system of elliptic equations.
 *
 * Requires the class has these static member variables:
 * - `size_t volume_dim`: The spatial dimension of tensors for this solution
 *
 * Require the class has these type aliases:
 * - `supported_tags`: A typelist of tags that can be retrieved from the
 * `variables` function.
 *
 * Requires the class has these member functions:
 * - `variables`: Returns a collection of variables at coordinates `x`. The
 * particular set of variables to be computed is specified as a `tmpl::list` of
 * tags that is supplied as the third `meta` argument. Must support any subset
 * and permutation of the `supported_tags`. When multiple variables are
 * requested, this function should make sure that precomputed data is shared
 * among the computation of these variables so it can be evaluated efficiently.
 *
 * This is an example implementation of a class that conforms to this protocol:
 *
 * \snippet AnalyticData/Test_Protocols.cpp elliptic_analytic_data_example
 */
struct AnalyticData {
  template <typename ConformingType>
  static constexpr bool is_conforming_v =
      analytic_data_protocols_detail::is_conforming_v<ConformingType>;
};

}  // namespace protocols
}  // namespace elliptic
