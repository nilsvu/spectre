// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {
namespace NumericalFluxes {

namespace detail {
template <typename FluxType, typename... AllFieldTags, typename... AllExtraTags,
          typename... Args, typename... PackageFieldTags,
          typename... PackageExtraTags>
void package_data_impl(
    const gsl::not_null<dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                               tmpl::list<AllExtraTags...>>*>
        packaged_data,
    const FluxType& flux_computer, tmpl::list<PackageFieldTags...> /*meta*/,
    tmpl::list<PackageExtraTags...> /*meta*/, const Args&... args) noexcept {
  flux_computer.package_data(
      make_not_null(&get<PackageFieldTags>(packaged_data->field_data))...,
      make_not_null(&get<PackageExtraTags>(packaged_data->extra_data))...,
      args...);
}

template <typename FluxType, typename... AllFieldTags, typename... AllExtraTags,
          typename... NormalDotNumericalFluxTypes, typename... PackageFieldTags,
          typename... PackageExtraTags>
void normal_dot_numerical_fluxes_impl(
    const FluxType& flux_computer,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_int,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_ext,
    tmpl::list<PackageFieldTags...> /*meta*/,
    tmpl::list<PackageExtraTags...> /*meta*/,
    const gsl::not_null<
        NormalDotNumericalFluxTypes*>... n_dot_num_fluxes) noexcept {
  flux_computer(n_dot_num_fluxes...,
                get<PackageFieldTags>(packaged_data_int.field_data)...,
                get<PackageExtraTags>(packaged_data_int.extra_data)...,
                get<PackageFieldTags>(packaged_data_ext.field_data)...,
                get<PackageExtraTags>(packaged_data_ext.extra_data)...);
}
}  // namespace detail

// @{
/// Helper function to unpack arguments when invoking the numerical flux
template <typename FluxType, typename... AllFieldTags, typename... AllExtraTags,
          typename... Args>
void package_data(
    const gsl::not_null<dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                               tmpl::list<AllExtraTags...>>*>
        packaged_data,
    const FluxType& flux_computer, const Args&... args) noexcept {
  detail::package_data_impl(packaged_data, flux_computer,
                            typename FluxType::package_field_tags{},
                            typename FluxType::package_extra_tags{}, args...);
}

template <typename FluxType, typename... AllFieldTags, typename... AllExtraTags,
          typename... NormalDotNumericalFluxTypes>
void normal_dot_numerical_fluxes(
    const FluxType& flux_computer,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_int,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_ext,
    const gsl::not_null<
        NormalDotNumericalFluxTypes*>... n_dot_num_fluxes) noexcept {
  detail::normal_dot_numerical_fluxes_impl(
      flux_computer, packaged_data_int, packaged_data_ext,
      typename FluxType::package_field_tags{},
      typename FluxType::package_extra_tags{}, n_dot_num_fluxes...);
}

template <typename FluxType, typename... AllFieldTags, typename... AllExtraTags,
          typename... NormalDotNumericalFluxTags>
void normal_dot_numerical_fluxes(
    const gsl::not_null<Variables<tmpl::list<NormalDotNumericalFluxTags...>>*>
        n_dot_num_fluxes,
    const FluxType& flux_computer,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_int,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_ext) noexcept {
  detail::normal_dot_numerical_fluxes_impl(
      flux_computer, packaged_data_int, packaged_data_ext,
      typename FluxType::package_field_tags{},
      typename FluxType::package_extra_tags{},
      make_not_null(&get<NormalDotNumericalFluxTags>(*n_dot_num_fluxes))...);
}
// @}

}  // namespace NumericalFluxes
}  // namespace dg
