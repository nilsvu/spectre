// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// Wrappers to zero-out Hamiltonian and lapse sources because the functions in
// `Xcts::` add contributions to them
void momentum_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_for_shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain) noexcept {
  std::fill(source_for_conformal_factor->begin(),
            source_for_conformal_factor->end(), 0.);
  std::fill(source_for_lapse_times_conformal_factor->begin(),
            source_for_lapse_times_conformal_factor->end(), 0.);
  Xcts::momentum_sources(
      source_for_conformal_factor, source_for_lapse_times_conformal_factor,
      source_for_shift, momentum_density, extrinsic_curvature_trace_gradient,
      conformal_factor, lapse_times_conformal_factor, shift,
      conformal_factor_gradient, lapse_times_conformal_factor_gradient,
      shift_strain);
}

void linearized_momentum_sources_wrapper(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_for_shift_correction,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift_correction,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        conformal_factor_gradient_correction,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient_correction,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        shift_strain_correction) noexcept {
  std::fill(source_for_conformal_factor_correction->begin(),
            source_for_conformal_factor_correction->end(), 0.);
  std::fill(source_for_lapse_times_conformal_factor_correction->begin(),
            source_for_lapse_times_conformal_factor_correction->end(), 0.);
  Xcts::linearized_momentum_sources(
      source_for_conformal_factor_correction,
      source_for_lapse_times_conformal_factor_correction,
      source_for_shift_correction, momentum_density,
      extrinsic_curvature_trace_gradient, conformal_factor,
      lapse_times_conformal_factor, shift, conformal_factor_gradient,
      lapse_times_conformal_factor_gradient, shift_strain,
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_correction, conformal_factor_gradient_correction,
      lapse_times_conformal_factor_gradient_correction,
      shift_strain_correction);
}

void test_equations(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(&Xcts::longitudinal_shift, "Equations",
                                    {"longitudinal_shift"}, {{{-1., 1.}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(&Xcts::hamiltonian_sources, "Equations",
                                    {"hamiltonian_sources"}, {{{-1., 1.}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(
      &Xcts::linearized_hamiltonian_sources, "Equations",
      {"linearized_hamiltonian_sources"}, {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(&Xcts::lapse_sources, "Equations",
                                    {"lapse_sources"}, {{{-1., 1.}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(&Xcts::linearized_lapse_sources,
                                    "Equations", {"linearized_lapse_sources"},
                                    {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &momentum_sources_wrapper, "Equations",
      {"shift_contribution_to_hamiltonian_sources",
       "shift_contribution_to_lapse_sources", "momentum_sources"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &linearized_momentum_sources_wrapper, "Equations",
      {"linearized_shift_contribution_to_hamiltonian_sources",
       "linearized_shift_contribution_to_lapse_sources",
       "linearized_momentum_sources"},
      {{{-1., 1.}}}, used_for_size);
}

template <Xcts::Equations EnabledEquations>
void test_computers(const DataVector& used_for_size) {
  CAPTURE(EnabledEquations);
  using system = Xcts::FirstOrderSystem<EnabledEquations, Xcts::Geometry::Flat>;
  using primal_fields = typename system::primal_fields;
  using auxiliary_fields = typename system::auxiliary_fields;
  using vars_tag = typename system::fields_tag;
  using VarsType = db::item_type<vars_tag>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<3>,
                                        Frame::Inertial>;
  using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag>;
  using FluxesComputer = typename system::fluxes;
  using SourcesComputer = typename system::sources;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  {
    INFO("Fluxes");
    const FluxesComputer fluxes_computer{};
    const auto computed_fluxes =
        elliptic::first_order_fluxes<3, primal_fields, auxiliary_fields>(
            vars, fluxes_computer);

    using argument_tags = typename FluxesComputer::argument_tags;
    auto box = db::create<db::AddSimpleTags<vars_tag, fluxes_tag>>(
        vars, make_with_value<db::item_type<fluxes_tag>>(
                  used_for_size, std::numeric_limits<double>::signaling_NaN()));
    tuples::tagged_tuple_from_typelist<typename VarsType::tags_list>
        vars_tuple{};
    tmpl::for_each<typename VarsType::tags_list>(
        [&vars, &vars_tuple](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(vars_tuple) = get<tag>(vars);
        });
    tuples::apply<auxiliary_fields>(
        [&fluxes_computer, &box](const auto&... aux_vars) {
          db::mutate_apply<db::wrap_tags_in<::Tags::Flux, primal_fields,
                                            tmpl::size_t<3>, Frame::Inertial>,
                           argument_tags>(fluxes_computer, make_not_null(&box),
                                          aux_vars...);
        },
        vars_tuple);
    tuples::apply<primal_fields>(
        [&fluxes_computer, &box](const auto&... primal_vars) {
          db::mutate_apply<db::wrap_tags_in<::Tags::Flux, auxiliary_fields,
                                            tmpl::size_t<3>, Frame::Inertial>,
                           argument_tags>(fluxes_computer, make_not_null(&box),
                                          primal_vars...);
        },
        vars_tuple);
    CHECK(computed_fluxes == get<fluxes_tag>(box));
  }
  {
    INFO("Sources");
    using argument_tags = typename SourcesComputer::argument_tags;
    tuples::tagged_tuple_from_typelist<argument_tags> args{};
    tmpl::for_each<argument_tags>(
        [&args, &generator, &dist, &used_for_size](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(args) = make_with_random_values<db::item_type<tag>>(
              make_not_null(&generator), make_not_null(&dist), used_for_size);
        });
    const auto computed_sources = tuples::apply(
        [&vars](const auto&... args) {
          return elliptic::first_order_sources<primal_fields, auxiliary_fields,
                                               SourcesComputer>(vars, args...);
        },
        args);

    auto box = tuples::apply(
        [&vars, &used_for_size](const auto&... args) {
          return db::create<
              tmpl::append<tmpl::list<vars_tag, sources_tag>, argument_tags>>(
              vars,
              make_with_value<db::item_type<sources_tag>>(
                  used_for_size, std::numeric_limits<double>::signaling_NaN()),
              args...);
        },
        args);
    tuples::tagged_tuple_from_typelist<typename VarsType::tags_list>
        vars_tuple{};
    tmpl::for_each<typename VarsType::tags_list>(
        [&vars, &vars_tuple](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(vars_tuple) = get<tag>(vars);
        });
    tuples::apply(
        [&box](const auto&... vars) {
          db::mutate_apply<db::wrap_tags_in<::Tags::Source, primal_fields>,
                           argument_tags>(SourcesComputer{},
                                          make_not_null(&box), vars...);
        },
        vars_tuple);
    tmpl::for_each<auxiliary_fields>([&vars, &box](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      db::mutate<::Tags::Source<tag>>(
          make_not_null(&box),
          [&vars](const auto aux_source) { *aux_source = get<tag>(vars); });
    });
    CHECK(computed_sources == get<sources_tag>(box));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"Elliptic/Systems/Xcts"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  test_equations(dv);
  CHECK_FOR_DATAVECTORS(
      test_computers,
      (Xcts::Equations::Hamiltonian, Xcts::Equations::HamiltonianAndLapse,
       Xcts::Equations::HamiltonianLapseAndShift));
}
