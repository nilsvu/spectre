// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Helper functions to test elliptic first-order systems

#pragma once

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestHelpers::elliptic {

template <typename System>
void test_first_order_fluxes_computer(
    const typename System::fluxes& fluxes_computer,
    const DataVector& used_for_size) {
  using FluxesComputer = typename System::fluxes;
  static constexpr size_t volume_dim = System::volume_dim;
  using vars_tag = typename System::fields_tag;
  using primal_fields = typename System::primal_fields;
  using auxiliary_fields = typename System::auxiliary_fields;
  using VarsType = typename vars_tag::type;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using argument_tags = typename FluxesComputer::argument_tags;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  // Generate random variables
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate fluxes from the variables with random arguments
  tuples::tagged_tuple_from_typelist<argument_tags> args{};
  tmpl::for_each<argument_tags>(
      [&args, &generator, &dist, &used_for_size](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(args) = make_with_random_values<typename tag::type>(
            make_not_null(&generator), make_not_null(&dist), used_for_size);
      });
  const auto computed_fluxes = tuples::apply(
      [&vars, &fluxes_computer](const auto&... expanded_args) {
        return ::elliptic::first_order_fluxes<volume_dim, primal_fields,
                                              auxiliary_fields>(
            vars, fluxes_computer, expanded_args...);
      },
      args);

  // Create a DataBox
  auto box = tuples::apply(
      [&vars, &used_for_size](const auto&... expanded_args) {
        return db::create<
            tmpl::append<tmpl::list<vars_tag, fluxes_tag>, argument_tags>>(
            vars,
            make_with_value<typename fluxes_tag::type>(
                used_for_size, std::numeric_limits<double>::signaling_NaN()),
            expanded_args...);
      },
      args);

  // Apply the fluxes computer to the DataBox
  tuples::tagged_tuple_from_typelist<typename VarsType::tags_list> vars_tuple{};
  tmpl::for_each<typename VarsType::tags_list>(
      [&vars, &vars_tuple](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(vars_tuple) = get<tag>(vars);
      });
  tuples::apply<auxiliary_fields>(
      [&fluxes_computer, &box](const auto&... auxiliary_vars) {
        db::mutate_apply<
            db::wrap_tags_in<::Tags::Flux, primal_fields,
                             tmpl::size_t<volume_dim>, Frame::Inertial>,
            argument_tags>(fluxes_computer, make_not_null(&box),
                           auxiliary_vars...);
      },
      vars_tuple);
  tuples::apply<primal_fields>(
      [&fluxes_computer, &box](const auto&... primal_vars) {
        db::mutate_apply<
            db::wrap_tags_in<::Tags::Flux, auxiliary_fields,
                             tmpl::size_t<volume_dim>, Frame::Inertial>,
            argument_tags>(fluxes_computer, make_not_null(&box),
                           primal_vars...);
      },
      vars_tuple);
  CHECK(computed_fluxes == get<fluxes_tag>(box));
}

template <typename System>
void test_first_order_sources_computer(const DataVector& used_for_size) {
  using SourcesComputer = typename System::sources;
  static constexpr size_t volume_dim = System::volume_dim;
  using vars_tag = typename System::fields_tag;
  using primal_fields = typename System::primal_fields;
  using auxiliary_fields = typename System::auxiliary_fields;
  using VarsType = typename vars_tag::type;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using FluxesType = typename fluxes_tag::type;
  using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag>;
  using argument_tags = typename SourcesComputer::argument_tags;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  // Generate random variables and fluxes
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto fluxes = make_with_random_values<FluxesType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate sources from the variables with random arguments
  tuples::tagged_tuple_from_typelist<argument_tags> args{};
  tmpl::for_each<argument_tags>(
      [&args, &generator, &dist, &used_for_size](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        get<tag>(args) = make_with_random_values<typename tag::type>(
            make_not_null(&generator), make_not_null(&dist), used_for_size);
      });
  const auto computed_sources = tuples::apply(
      [&vars, &fluxes](const auto&... expanded_args) {
        return ::elliptic::first_order_sources<
            volume_dim, primal_fields, auxiliary_fields, SourcesComputer>(
            vars, fluxes, expanded_args...);
      },
      args);

  // Create a DataBox
  auto box = tuples::apply(
      [&vars, &fluxes, &used_for_size](const auto&... expanded_args) {
        return db::create<tmpl::append<
            tmpl::list<vars_tag, fluxes_tag, sources_tag>, argument_tags>>(
            vars, fluxes,
            make_with_value<typename sources_tag::type>(
                used_for_size, std::numeric_limits<double>::signaling_NaN()),
            expanded_args...);
      },
      args);

  // Apply the sources computer to the DataBox
  tuples::tagged_tuple_from_typelist<
      tmpl::append<primal_fields,
                   db::wrap_tags_in<::Tags::Flux, primal_fields,
                                    tmpl::size_t<volume_dim>, Frame::Inertial>>>
      vars_tuple{};
  tmpl::for_each<primal_fields>([&vars, &fluxes, &vars_tuple](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    using flux_tag =
        ::Tags::Flux<tag, tmpl::size_t<volume_dim>, Frame::Inertial>;
    get<tag>(vars_tuple) = get<tag>(vars);
    get<flux_tag>(vars_tuple) = get<flux_tag>(fluxes);
  });
  tmpl::for_each<auxiliary_fields>([&vars, &box](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    db::mutate<::Tags::Source<tag>>(
        make_not_null(&box),
        [&vars](const auto aux_source) { *aux_source = get<tag>(vars); });
  });
  tuples::apply(
      [&box](const auto&... expanded_vars) {
        db::mutate_apply<
            db::wrap_tags_in<::Tags::Source, typename vars_tag::tags_list>,
            argument_tags>(SourcesComputer{}, make_not_null(&box),
                           expanded_vars...);
      },
      vars_tuple);
  CHECK(computed_sources == get<sources_tag>(box));
}

template <typename Tag>
struct Correction : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct Corrected : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// Test the `System` has a `linearized_system` and that it is actually the
/// linearization of its nonlinear fluxes and sources to the order given by
/// the `correction_magnitude`
template <typename System>
void test_linearization(const double correction_magnitude,
                        const DataVector& used_for_size) {
  CAPTURE(correction_magnitude);

  using system = System;
  using linearized_system = typename system::linearized_system;
  static constexpr size_t Dim = system::volume_dim;

  using fields_tag = typename system::fields_tag;
  using Fields = typename fields_tag::type;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, fields_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using Fluxes = typename fluxes_tag::type;

  using primal_fields = typename system::primal_fields;
  using auxiliary_fields = typename system::auxiliary_fields;

  using fields_correction_tag = db::add_tag_prefix<Correction, fields_tag>;
  using fluxes_correction_tag =
      db::add_tag_prefix<::Tags::Flux, fields_correction_tag, tmpl::size_t<Dim>,
                         Frame::Inertial>;
  using fields_corrected_tag = db::add_tag_prefix<Corrected, fields_tag>;
  using fluxes_corrected_tag =
      db::add_tag_prefix<::Tags::Flux, fields_corrected_tag, tmpl::size_t<Dim>,
                         Frame::Inertial>;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(0.5, 1.);
  std::uniform_real_distribution<> dist_eps(-correction_magnitude,
                                            correction_magnitude);
  Approx custom_approx =
      Approx::custom().epsilon(correction_magnitude).scale(1.);

  const auto background_fields =
      make_with_random_values<Variables<typename system::background_fields>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto fields = make_with_random_values<Fields>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto fluxes = make_with_random_values<Fluxes>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto fields_correction =
      make_with_random_values<typename fields_correction_tag::type>(
          make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  const auto fluxes_correction =
      make_with_random_values<typename fluxes_correction_tag::type>(
          make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  const typename fields_corrected_tag::type fields_corrected{fields +
                                                             fields_correction};
  const typename fluxes_corrected_tag::type fluxes_corrected{fluxes +
                                                             fluxes_correction};

  auto box = db::create<
      db::AddSimpleTags<::Tags::Variables<typename system::background_fields>,
                        fields_tag, fluxes_tag, fields_correction_tag,
                        fluxes_correction_tag, fields_corrected_tag,
                        fluxes_corrected_tag>,
      db::AddComputeTags<
          ::elliptic::Tags::FirstOrderSourcesCompute<
              Dim, typename system::sources, fields_tag, primal_fields,
              auxiliary_fields>,
          ::elliptic::Tags::FirstOrderSourcesCompute<
              Dim, typename system::sources, fields_corrected_tag,
              db::wrap_tags_in<Corrected, primal_fields>,
              db::wrap_tags_in<Corrected, auxiliary_fields>>,
          ::elliptic::Tags::FirstOrderSourcesCompute<
              Dim, typename linearized_system::sources, fields_correction_tag,
              db::wrap_tags_in<Correction, primal_fields>,
              db::wrap_tags_in<Correction, auxiliary_fields>>>>(
      background_fields, fields, fluxes, fields_correction, fluxes_correction,
      fields_corrected, fluxes_corrected);

  const typename db::add_tag_prefix<::Tags::Source, fields_correction_tag>::type
      sources_diff{
          get<db::add_tag_prefix<::Tags::Source, fields_corrected_tag>>(box) -
          get<db::add_tag_prefix<::Tags::Source, fields_tag>>(box)};
  const auto& sources_diff_linear =
      get<db::add_tag_prefix<::Tags::Source, fields_correction_tag>>(box);
  CHECK_VARIABLES_CUSTOM_APPROX(sources_diff, sources_diff_linear,
                                custom_approx);
}

}  // namespace TestHelpers::elliptic
