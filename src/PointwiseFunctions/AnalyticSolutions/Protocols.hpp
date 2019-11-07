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

namespace evolution {
namespace protocols {

/*!
 * \ingroup ProtocolsGroup
 * \brief An analytic solution to a system of evolution equations.
 *
 * Requires the class has this metaprogramming interface:
 * - `size_t volume_dim`: The spatial dimension of tensors for this solution
 * - TODO: We should probably add a `supported_tags` alias, or even the `system`
 *
 * Requires the class has these member functions:
 * - `variables`: Returns a collection of variables at coordinates `x` and time
 * `t`. The particular set of variables to be computed is specified as a
 * `tmpl::list` of tags that is supplied as the third `meta` argument. When
 * multiple variables are requested, this function should make sure that
 * precomputed data is shared among the computation of these variables so it
 * can be evaluated efficiently.
 *
 * This is an example implementation of a class that conforms to this protocol:
 *
 * \snippet AnalyticSolutions/Test_Protocols.cpp evolution_analytic_sol_example
 */
struct AnalyticSolution {
 private:
  CREATE_HAS(volume_dim)
  CREATE_IS_CALLABLE(variables)

  template <typename ConformingType>
  struct IsVariablesCallable
      : is_variables_callable_r_t<
            tuples::TaggedTuple<>, ConformingType,
            tnsr::I<DataVector, ConformingType::volume_dim>, double,
            tmpl::list<>> {};

 public:
  template <typename ConformingType>
  static constexpr bool is_conforming_v =
      std::conditional_t<has_volume_dim_v<ConformingType, size_t>,
                         IsVariablesCallable<ConformingType>,
                         std::false_type>::value;
};

}  // namespace protocols
}  // namespace evolution

namespace elliptic {
namespace protocols {

/*!
 * \ingroup ProtocolsGroup
 * \brief An analytic solution to a system of elliptic equations.
 *
 * Requires the class has this metaprogramming interface:
 * - `size_t volume_dim`: The spatial dimension of tensors for this solution
 *
 * Requires the class has these member functions:
 * - `variables`: Returns a collection of variables at coordinates `x`.
 * The particular set of variables to be computed is specified as a
 * `tmpl::list` of tags that is supplied as the third `meta` argument. When
 * multiple variables are requested, this function should make sure that to
 * share precomputed data among the computation of these variables so it can be
 * evaluated efficiently.
 *
 * This is an example implementation of a class that conforms to this protocol:
 *
 * \snippet AnalyticSolutions/Test_Protocols.cpp elliptic_analytic_sol_example
 */
struct AnalyticSolution {
 private:
  CREATE_HAS(volume_dim)
  CREATE_IS_CALLABLE(variables)

  template <typename ConformingType>
  struct IsVariablesCallable
      : is_variables_callable_r_t<
            tuples::TaggedTuple<>, ConformingType,
            tnsr::I<DataVector, ConformingType::volume_dim>,
            tmpl::list<>> {};

 public:
  template <typename ConformingType>
  static constexpr bool is_conforming_v =
      std::conditional_t<has_volume_dim_v<ConformingType, size_t>,
                         IsVariablesCallable<ConformingType>,
                         std::false_type>::value;
};

}  // namespace protocols
}  // namespace elliptic
