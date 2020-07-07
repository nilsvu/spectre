// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"

/// \cond
template <size_t Dim>
struct DomainCreator;
/// \endcond

namespace TestHelpers::Poisson::dg {

namespace helpers = TestHelpers::elliptic::dg;

using field_tag = ::Poisson::Tags::Field;
template <size_t Dim>
using field_gradient_tag =
    ::Tags::deriv<::Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>;
using operator_applied_to_field_tag = helpers::OperatorAppliedTo<field_tag>;
template <size_t Dim>
using operator_applied_to_field_gradient_tag =
    helpers::OperatorAppliedTo<field_gradient_tag<Dim>>;

template <size_t Dim>
using Vars = Variables<tmpl::list<field_tag, field_gradient_tag<Dim>>>;
template <size_t Dim>
using OperatorVars = Variables<db::wrap_tags_in<helpers::OperatorAppliedTo,
                                                typename Vars<Dim>::tags_list>>;

template <size_t Dim>
OperatorVars<Dim> apply_first_order_operator(
    const ElementId<Dim>& element_id,
    const helpers::ElementArray<Dim>& dg_elements,
    const std::unordered_map<ElementId<Dim>, Vars<Dim>>& all_vars,
    double penalty_parameter);

}  // namespace TestHelpers::Poisson::dg
