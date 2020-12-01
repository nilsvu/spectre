// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/CoordinateMapHelpers.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Tag>
struct Step : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

struct PrimalField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AuxiliaryField : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

struct AnArgument : db::SimpleTag {
  using type = double;
};

template <size_t Dim>
struct Fluxes {
  using argument_tags = tmpl::list<AnArgument>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const double an_argument,
      const tnsr::i<DataVector, Dim>& auxiliary_field) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_field->get(d) = auxiliary_field.get(d) * an_argument;
    }
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_aux_field,
      const double an_argument, const Scalar<DataVector>& field) {
    std::fill(flux_for_aux_field->begin(), flux_for_aux_field->end(), 0.);
    for (size_t d = 0; d < Dim; d++) {
      flux_for_aux_field->get(d, d) = get(field) * an_argument;
    }
  }
};

struct Sources {
  using argument_tags = tmpl::list<AnArgument>;
  template <size_t Dim>
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_field,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> /*source_for_aux_field*/,
      const double an_argument, const Scalar<DataVector>& field,
      const tnsr::I<DataVector, Dim>& /*field_flux*/) {
    get(*source_for_field) = get(field) * square(an_argument);
  }
};

template <size_t Dim>
void test_fluxes_and_sources() {
  // Construct some field data
  static constexpr size_t num_points = 3;
  using primal_fields = tmpl::list<PrimalField>;
  using auxiliary_fields = tmpl::list<AuxiliaryField<Dim>>;
  using all_fields = tmpl::append<primal_fields, auxiliary_fields>;
  Variables<all_fields> vars{num_points};
  get(get<PrimalField>(vars)) = DataVector{num_points, 2.};
  for (size_t d = 0; d < Dim; d++) {
    get<AuxiliaryField<Dim>>(vars).get(d) = DataVector{num_points, d + 1.};
  }

  // Test fluxes
  const Fluxes<Dim> fluxes_computer{};
  Variables<db::wrap_tags_in<::Tags::Flux, all_fields, tmpl::size_t<Dim>,
                             Frame::Inertial>>
      fluxes{num_points};
  elliptic::first_order_fluxes<Dim, primal_fields, auxiliary_fields>(
      make_not_null(&fluxes), vars, fluxes_computer, 3.);
  // Check return-by-ref and return-by-value functions are equal
  CHECK(fluxes ==
        elliptic::first_order_fluxes<Dim, primal_fields, auxiliary_fields>(
            vars, fluxes_computer, 3.));
  // Check computed values
  for (size_t d = 0; d < Dim; d++) {
    CHECK(get<::Tags::Flux<PrimalField, tmpl::size_t<Dim>, Frame::Inertial>>(
              fluxes)
              .get(d) == DataVector{num_points, 3. * (d + 1.)});
    CHECK(get<::Tags::Flux<AuxiliaryField<Dim>, tmpl::size_t<Dim>,
                           Frame::Inertial>>(fluxes)
              .get(d, d) == DataVector{num_points, 6.});
    for (size_t i0 = 0; i0 < Dim; i0++) {
      if (i0 != d) {
        CHECK(get<::Tags::Flux<AuxiliaryField<Dim>, tmpl::size_t<Dim>,
                               Frame::Inertial>>(fluxes)
                  .get(d, i0) == DataVector{num_points, 0.});
      }
    }
  }

  // Test sources
  Variables<db::wrap_tags_in<::Tags::Source, all_fields>> sources{num_points};
  elliptic::first_order_sources<Dim, primal_fields, auxiliary_fields, Sources>(
      make_not_null(&sources), vars, fluxes, 3.);
  // Check return-by-ref and return-by-value functions are equal
  CHECK(sources ==
        elliptic::first_order_sources<Dim, primal_fields, auxiliary_fields,
                                      Sources>(vars, fluxes, 3.));
  // Check computed values
  CHECK(get(get<::Tags::Source<PrimalField>>(sources)) ==
        DataVector{num_points, 18.});
  for (size_t d = 0; d < Dim; d++) {
    CHECK(get<::Tags::Source<AuxiliaryField<Dim>>>(sources).get(d) ==
          get<AuxiliaryField<Dim>>(vars).get(d));
  }
}

template <size_t Dim>
void test_first_order_operator(const DataVector& used_for_size) {
  using vars_tag =
      Tags::Variables<tmpl::list<PrimalField, AuxiliaryField<Dim>>>;
  using step_tag = db::add_tag_prefix<Step, vars_tag>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag>;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  {
    INFO("First-order operator from fluxes and sources");
    // Compute operator for random values
    const auto div_fluxes =
        make_with_random_values<typename div_fluxes_tag::type>(
            nn_generator, nn_dist, used_for_size);
    const auto sources = make_with_random_values<typename sources_tag::type>(
        nn_generator, nn_dist, used_for_size);
    auto step = make_with_value<typename step_tag::type>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    elliptic::first_order_operator(make_not_null(&step), div_fluxes, sources);

    CHECK(get(get<Step<PrimalField>>(step)) ==
          -get(get<::Tags::div<::Tags::Flux<PrimalField, tmpl::size_t<Dim>,
                                            Frame::Inertial>>>(div_fluxes)) +
              get(get<::Tags::Source<PrimalField>>(sources)));
    for (size_t d = 0; d < Dim; d++) {
      CHECK(
          get<Step<AuxiliaryField<Dim>>>(step).get(d) ==
          -get<::Tags::div<::Tags::Flux<AuxiliaryField<Dim>, tmpl::size_t<Dim>,
                                        Frame::Inertial>>>(div_fluxes)
                  .get(d) +
              get<::Tags::Source<AuxiliaryField<Dim>>>(sources).get(d));
    }

    // Check the mutating DataBox invokable
    auto box =
        db::create<db::AddSimpleTags<div_fluxes_tag, sources_tag, step_tag>>(
            div_fluxes, sources,
            make_with_value<typename step_tag::type>(
                used_for_size, std::numeric_limits<double>::signaling_NaN()));
    db::mutate_apply<elliptic::FirstOrderOperator<Dim, Step, vars_tag>>(
        make_not_null(&box));
    CHECK(get<step_tag>(box) == step);
  }
  {
    INFO("First-order operator from vars");
    Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const auto coord_map = []() {
      using Affine = domain::CoordinateMaps::Affine;
      if constexpr (Dim == 1) {
        return domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine>{
            {-1., 1., 0., 1.}};
      } else if constexpr (Dim == 2) {
        using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
        return domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>{
            {{-1., 1., 0., 1.}, {-1., 1., 0., 1.}}};
      } else if constexpr (Dim == 3) {
        using Affine3D =
            domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
        return domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine3D>{
            {{-1., 1., 0., 1.}, {-1., 1., 0., 1.}, {-1., 1., 0., 1.}}};
      }
    }();
    const auto inv_jacobian = coord_map.inv_jacobian(logical_coordinates(mesh));
    const double arg = 2.;
    const Fluxes<Dim> fluxes_computer{};
    const auto vars = make_with_random_values<typename vars_tag::type>(
        nn_generator, nn_dist, num_points);
    typename fluxes_tag::type fluxes{};
    typename div_fluxes_tag::type div_fluxes{};
    typename step_tag::type operator_applied_to_vars{};
    elliptic::first_order_operator<tmpl::list<PrimalField>,
                                   tmpl::list<AuxiliaryField<Dim>>, Sources>(
        make_not_null(&operator_applied_to_vars), make_not_null(&fluxes),
        make_not_null(&div_fluxes), vars, mesh, inv_jacobian, fluxes_computer,
        std::make_tuple(arg), std::make_tuple(arg));
    typename fluxes_tag::type expected_fluxes{};
    typename div_fluxes_tag::type expected_div_fluxes{};
    typename sources_tag::type expected_sources{};
    typename step_tag::type expected_operator_applied_to_vars{};
    elliptic::first_order_fluxes<Dim, tmpl::list<PrimalField>,
                                 tmpl::list<AuxiliaryField<Dim>>>(
        make_not_null(&expected_fluxes), vars, fluxes_computer, arg);
    CHECK_VARIABLES_APPROX(fluxes, expected_fluxes);
    divergence(make_not_null(&expected_div_fluxes), expected_fluxes, mesh,
               inv_jacobian);
    CHECK_VARIABLES_APPROX(div_fluxes, expected_div_fluxes);
    elliptic::first_order_sources<Dim, tmpl::list<PrimalField>,
                                  tmpl::list<AuxiliaryField<Dim>>, Sources>(
        make_not_null(&expected_sources), vars, expected_fluxes, arg);
    elliptic::first_order_operator(
        make_not_null(&expected_operator_applied_to_vars), expected_div_fluxes,
        expected_sources);
    CHECK_VARIABLES_APPROX(operator_applied_to_vars,
                           expected_operator_applied_to_vars);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.FirstOrderOperator", "[Unit][Elliptic]") {
  test_fluxes_and_sources<1>();
  test_fluxes_and_sources<2>();
  test_fluxes_and_sources<3>();

  const DataVector used_for_size{5};
  test_first_order_operator<1>(used_for_size);
  test_first_order_operator<2>(used_for_size);
  test_first_order_operator<3>(used_for_size);
}
