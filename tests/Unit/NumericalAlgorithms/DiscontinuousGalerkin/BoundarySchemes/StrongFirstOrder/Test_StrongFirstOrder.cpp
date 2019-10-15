// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeString.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace {

using TemporalId = int;

template <typename Tag>
struct BoundaryContribution : db::SimpleTag, db::PrefixTag {
  using tag = Tag;
  using type = db::item_type<Tag>;
};

struct TemporalIdTag : db::SimpleTag {
  using type = TemporalId;
  template <typename Tag>
  using step_prefix = BoundaryContribution<Tag>;
};

struct SomeField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using variables_tag = ::Tags::Variables<tmpl::list<SomeField>>;

struct NumericalFlux {
  using argument_tags = tmpl::list<TemporalIdTag, SomeField>;
  using volume_tags = tmpl::list<TemporalIdTag>;
  using package_field_tags = tmpl::list<SomeField>;
  using package_extra_tags = tmpl::list<TemporalIdTag>;
  static void package_data(
      const gsl::not_null<Scalar<DataVector>*> packaged_field,
      const gsl::not_null<int*> packaged_extra_data, const int& temporal_id,
      const Scalar<DataVector>& field) noexcept {
    *packaged_field = field;
    *packaged_extra_data = temporal_id;
  }
  void operator()(const gsl::not_null<Scalar<DataVector>*> n_dot_numerical_flux,
                  const Scalar<DataVector>& field_int,
                  const TemporalId& time_int,
                  const Scalar<DataVector>& field_ext,
                  const TemporalId& time_ext) const noexcept {
    CHECK(time_int == time_ext);
    // A simple central flux
    get(*n_dot_numerical_flux) = 0.5 * (get(field_int) + get(field_ext));
  }
};

// A flux used in earlier versions of this test to make sure the results didn't
// change (see history of `Test_MortarHelpers.cpp`)
struct RefinementTestsNumericalFlux {
  DataVector answer;

  using argument_tags = tmpl::list<SomeField>;
  using package_field_tags = tmpl::list<SomeField>;
  using package_extra_tags = tmpl::list<>;
  void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux,
                  const Scalar<DataVector>& local_var,
                  const Scalar<DataVector>& remote_var) const noexcept {
    CHECK(get(local_var) == DataVector{1., 2., 3.});
    CHECK(get(remote_var) == DataVector{6., 5., 4.});
    get(*numerical_flux) = answer;
  }
};

template <typename NumericalFluxType>
struct NumericalFluxTag : db::SimpleTag {
  using type = NumericalFluxType;
};

// We can probably move this to `src`
template <size_t Dim>
using MortarId = std::pair<Direction<Dim>, ElementId<Dim>>;

// Helper function to combine local and remote boundary data to mortar data
template <typename BoundaryScheme,
          typename BoundaryData = typename BoundaryScheme::BoundaryData>
auto make_mortar_data(const MortarId<BoundaryScheme::volume_dim>& mortar_id,
                      const TemporalId& time, BoundaryData&& interior_data,
                      BoundaryData exterior_data) noexcept {
  db::item_type<typename BoundaryScheme::mortar_data_tag> mortar_data{};
  mortar_data.local_insert(time, std::move(interior_data));
  mortar_data.remote_insert(time, std::move(exterior_data));
  return db::item_type<::Tags::Mortars<typename BoundaryScheme::mortar_data_tag,
                                       BoundaryScheme::volume_dim>>{
      {mortar_id, std::move(mortar_data)}};
}

// Helper function that returns the slice of the boundary contributions that
// correponds to the face we are considering, and checks that the rest of the
// data is zero.
template <size_t Dim>
Scalar<DataVector> extract_on_face(
    const gsl::not_null<
        db::item_type<db::add_tag_prefix<BoundaryContribution, variables_tag>>*>
        all_boundary_contributions,
    const Mesh<Dim>& volume_mesh, const size_t face_dim) noexcept {
  auto& field_in_volume =
      get<BoundaryContribution<SomeField>>(*all_boundary_contributions);
  Scalar<DataVector> field_on_face{
      volume_mesh.slice_away(face_dim).number_of_grid_points(),
      std::numeric_limits<double>::signaling_NaN()};
  for (SliceIterator slice_index{volume_mesh.extents(), face_dim,
                                 volume_mesh.extents(face_dim) - 1};
       slice_index; ++slice_index) {
    auto& point_contribution =
        get(field_in_volume)[slice_index.volume_offset()];
    get(field_on_face)[slice_index.slice_offset()] = point_contribution;
    // Zero the point contribution on the slice so we can check below that all
    // points away from the slice are still zero
    point_contribution = 0.;
  }
  CHECK(field_in_volume ==
        Scalar<DataVector>{volume_mesh.number_of_grid_points(), 0.});
  return field_on_face;
}

// Helper function to compare a simple setup to the Python implementation
template <size_t Dim, typename BoundaryScheme, size_t NumPointsPerDim>
Scalar<DataVector> lifted_boundary_contribution(
    const Scalar<DataVector>& field_int,
    const Scalar<DataVector>& field_ext) noexcept {
  // Setup a volume mesh
  const size_t num_points_per_dim = NumPointsPerDim;
  CAPTURE(num_points_per_dim);
  const Mesh<Dim> mesh{num_points_per_dim, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  const TemporalId time{1};
  // Setup a mortar
  const auto mortar_mesh = mesh.slice_away(0);
  const size_t mortar_num_points = mortar_mesh.number_of_grid_points();
  const MortarId<Dim> mortar_id{Direction<Dim>::upper_xi(), ElementId<Dim>{1}};
  std::array<Spectral::MortarSize, Dim - 1> mortar_size{};
  mortar_size.fill(Spectral::MortarSize::Full);
  // Make sure the input fields are of the correct size
  ASSERT(get(field_int).size() == mortar_num_points &&
             get(field_ext).size() == mortar_num_points,
         "The input fields have "
             << get(field_int).size() << " points but should have "
             << mortar_num_points
             << "points since they represent data on the boundary.");
  // Construct data on either side of the mortar
  const auto make_boundary_data = [&mortar_num_points](
      const Scalar<DataVector>& field,
      const Direction<Dim>& direction) noexcept {
    typename BoundaryScheme::BoundaryData boundary_data{mortar_num_points};
    get<SomeField>(boundary_data.field_data) = field;
    get(get<::Tags::NormalDotFlux<SomeField>>(boundary_data.field_data)) =
        direction.sign() * get(field);
    get<::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>(
        boundary_data.extra_data) = Scalar<DataVector>{mortar_num_points, 1.};
    get<TemporalIdTag>(boundary_data.extra_data) = time;
    return boundary_data;
  };
  auto all_mortar_data = make_mortar_data<BoundaryScheme>(
      mortar_id, time,
      make_boundary_data(field_int, Direction<Dim>::upper_xi()),
      make_boundary_data(field_ext, Direction<Dim>::lower_xi()));
  // Apply boundary scheme
  db::item_type<db::add_tag_prefix<BoundaryContribution, variables_tag>>
      all_boundary_contributions{num_points, 0.};
  BoundaryScheme::apply(make_not_null(&all_boundary_contributions),
                        make_not_null(&all_mortar_data), mesh,
                        {{mortar_id, mortar_mesh}}, {{mortar_id, mortar_size}},
                        NumericalFlux{});
  return extract_on_face(make_not_null(&all_boundary_contributions), mesh,
                         mortar_id.first.dimension());
}

template <size_t Dim>
void test_strong_first_order() noexcept {
  CAPTURE(Dim);
  using boundary_scheme = dg::BoundarySchemes::StrongFirstOrder<
      Dim, variables_tag, NumericalFluxTag<NumericalFlux>, TemporalIdTag>;
  using BoundaryData = typename boundary_scheme::BoundaryData;
  using mortar_data_tag = typename boundary_scheme::mortar_data_tag;
  using all_normal_dot_fluxes_tag = ::Tags::Interface<
      ::Tags::InternalDirections<Dim>,
      db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>>;
  using all_magnitudes_of_face_normals_tag =
      ::Tags::Interface<::Tags::InternalDirections<Dim>,
                        ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>;
  {
    INFO("Collect boundary data from a DataBox");
    const TemporalId time = 1;
    // Create a DataBox that holds the arguments for the numerical flux plus
    // those for the strong first-order boundary scheme
    const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const ElementId<Dim> element_id{0};
    const auto face_direction = Direction<Dim>::upper_xi();
    const ElementId<Dim> neighbor_id{1};
    const Element<Dim> element{element_id,
                               {{face_direction, {{neighbor_id}, {}}}}};
    const size_t num_points_on_face =
        mesh.slice_away(face_direction.dimension()).number_of_grid_points();
    db::item_type<db::add_tag_prefix<::Tags::NormalDotFlux, variables_tag>>
        normal_dot_fluxes{num_points_on_face};
    get<::Tags::NormalDotFlux<SomeField>>(normal_dot_fluxes) =
        Scalar<DataVector>{num_points_on_face, 3.};
    const auto box = db::create<
        db::AddSimpleTags<NumericalFluxTag<NumericalFlux>, TemporalIdTag,
                          SomeField, ::Tags::Mesh<Dim>, ::Tags::Element<Dim>,
                          all_normal_dot_fluxes_tag,
                          all_magnitudes_of_face_normals_tag>,
        db::AddComputeTags<
            ::Tags::InternalDirections<Dim>,
            ::Tags::InterfaceCompute<::Tags::InternalDirections<Dim>,
                                     ::Tags::Direction<Dim>>,
            ::Tags::InterfaceCompute<::Tags::InternalDirections<Dim>,
                                     ::Tags::InterfaceMesh<Dim>>,
            ::Tags::Slice<::Tags::InternalDirections<Dim>, SomeField>>>(
        NumericalFlux{}, time, Scalar<DataVector>{num_points, 2.}, mesh,
        element,
        db::item_type<all_normal_dot_fluxes_tag>{
            {face_direction, std::move(normal_dot_fluxes)}},
        db::item_type<all_magnitudes_of_face_normals_tag>{
            {face_direction, Scalar<DataVector>{num_points_on_face, 1.5}}});
    // Collect the boundary data needed by the boundary scheme
    const auto all_boundary_data =
        interface_apply<typename boundary_scheme::boundary_data_computer,
                        ::Tags::InternalDirections<Dim>>(box);
    // Make sure the collected boundary data is what we expect
    const auto check_face = [&all_boundary_data, &mesh,
                             &time](const Direction<Dim>& direction) {
      const size_t num_points_on_face =
          mesh.slice_away(direction.dimension()).number_of_grid_points();
      const auto& boundary_data = all_boundary_data.at(direction);
      CHECK(get<SomeField>(boundary_data.field_data) ==
            Scalar<DataVector>{num_points_on_face, 2.});
      CHECK(get<::Tags::NormalDotFlux<SomeField>>(boundary_data.field_data) ==
            Scalar<DataVector>{num_points_on_face, 3.});
      CHECK(get<TemporalIdTag>(boundary_data.extra_data) == time);
      CHECK(get<::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>(
                boundary_data.extra_data) ==
            Scalar<DataVector>{num_points_on_face, 1.5});
    };
    check_face(face_direction);
  }
  {
    INFO("Compare to Python implementation");
    constexpr size_t num_points_per_dim = 5;
    const DataVector used_for_size_on_face{pow<Dim - 1>(num_points_per_dim)};
    pypp::check_with_random_values<1>(
        &lifted_boundary_contribution<Dim, boundary_scheme, num_points_per_dim>,
        "StrongFirstOrder", "lifted_boundary_contribution", {{{-1., 1.}}},
        used_for_size_on_face);
  }
  {
    // This part only tests that the boundary scheme can be applied to mutate a
    // DataBox. It can be replaced by a generic test that checks the struct
    // conforms to the interface that `mutate_apply` expects (once we have such
    // a test).
    INFO("Apply to DataBox");
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(-1., 1.);
    const auto nn_generator = make_not_null(&generator);
    const auto nn_dist = make_not_null(&dist);
    // Setup a volume mesh
    const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    const TemporalId time{1};
    // Setup a mortar
    const auto mortar_mesh = mesh.slice_away(0);
    const size_t mortar_num_points = mortar_mesh.number_of_grid_points();
    const MortarId<Dim> mortar_id{Direction<Dim>::upper_xi(),
                                  ElementId<Dim>{1}};
    std::array<Spectral::MortarSize, Dim - 1> mortar_size{};
    mortar_size.fill(Spectral::MortarSize::Full);
    const DataVector used_for_size_on_mortar{mortar_num_points};
    // Fake some boundary data
    const auto make_boundary_data =
        [&used_for_size_on_mortar, &nn_generator, &nn_dist ]() noexcept {
      BoundaryData boundary_data{used_for_size_on_mortar.size()};
      get<SomeField>(boundary_data.field_data) =
          make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist,
                                                      used_for_size_on_mortar);
      get<::Tags::NormalDotFlux<SomeField>>(boundary_data.field_data) =
          make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist,
                                                      used_for_size_on_mortar);
      get<::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>(
          boundary_data.extra_data) =
          make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist,
                                                      used_for_size_on_mortar);
      get<TemporalIdTag>(boundary_data.extra_data) = time;
      return boundary_data;
    };
    auto all_mortar_data = make_mortar_data<boundary_scheme>(
        mortar_id, time, make_boundary_data(), make_boundary_data());
    // Assemble a DataBox and test
    db::item_type<db::add_tag_prefix<BoundaryContribution, variables_tag>>
        boundary_contributions{num_points, 0.};
    auto box = db::create<db::AddSimpleTags<
        NumericalFluxTag<NumericalFlux>, ::Tags::Mesh<Dim>,
        ::Tags::Mortars<mortar_data_tag, Dim>,
        ::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>,
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
        db::add_tag_prefix<BoundaryContribution, variables_tag>>>(
        NumericalFlux{}, mesh, all_mortar_data,
        db::item_type<::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>>{
            {mortar_id, mortar_mesh}},
        db::item_type<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>{
            {mortar_id, mortar_size}},
        std::move(boundary_contributions));
    db::mutate_apply<boundary_scheme>(make_not_null(&box));
    db::item_type<db::add_tag_prefix<BoundaryContribution, variables_tag>>
        expected_boundary_contributions{num_points, 0.};
    boundary_scheme::apply(make_not_null(&expected_boundary_contributions),
                           make_not_null(&all_mortar_data), mesh,
                           {{mortar_id, mortar_mesh}},
                           {{mortar_id, mortar_size}}, NumericalFlux{});
    const auto& mutated_boundary_contributions =
        get<db::add_tag_prefix<BoundaryContribution, variables_tag>>(box);
    CHECK_VARIABLES_APPROX(mutated_boundary_contributions,
                           expected_boundary_contributions);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.BoundarySchemes.StrongFirstOrder",
                  "[Unit][NumericalAlgorithms]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/"
      "StrongFirstOrder/");
  test_strong_first_order<1>();
  test_strong_first_order<2>();
  test_strong_first_order<3>();

  {
    // This test was carried over from Test_MortarHelpers.cpp
    INFO("p-refinement");
    static constexpr size_t Dim = 2;
    using boundary_scheme = dg::BoundarySchemes::StrongFirstOrder<
        Dim, variables_tag, NumericalFluxTag<RefinementTestsNumericalFlux>,
        TemporalIdTag>;
    RefinementTestsNumericalFlux numerical_flux{{0., 3., 0.}};
    const Mesh<Dim> mesh{{{4, 2}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    // Setup a mortar
    const MortarId<Dim> mortar_id{Direction<Dim>::upper_xi(),
                                  ElementId<Dim>{1}};
    const size_t slice_dim = mortar_id.first.dimension();
    const auto face_mesh = mesh.slice_away(slice_dim);
    const size_t face_num_points = face_mesh.number_of_grid_points();
    // The face has 2 grid points, but we make a mortar mesh with 3 grid points,
    // so this test includes a projection from a p-refined mortar mesh.
    const Mesh<Dim - 1> mortar_mesh{3, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
    const size_t mortar_num_points = mortar_mesh.number_of_grid_points();
    const size_t extent_perpendicular_to_boundary = mesh.extents(slice_dim);
    const std::array<Spectral::MortarSize, Dim - 1> mortar_size{
        {Spectral::MortarSize::Full}};
    // Construct boundary data
    const Scalar<DataVector> magnitude_of_face_normal{2., 5.};
    typename boundary_scheme::BoundaryData interior_data{mortar_num_points};
    get(get<SomeField>(interior_data.field_data)) = DataVector{1., 2., 3.};
    get(get<Tags::NormalDotFlux<SomeField>>(interior_data.field_data)) =
        DataVector{-3., 0., 3.};
    get<::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>(
        interior_data.extra_data) = magnitude_of_face_normal;
    typename boundary_scheme::BoundaryData exterior_data{mortar_num_points};
    get(get<SomeField>(exterior_data.field_data)) = DataVector{6., 5., 4.};
    auto all_mortar_data = make_mortar_data<boundary_scheme>(
        mortar_id, 0, std::move(interior_data), std::move(exterior_data));
    // Apply boundary scheme
    db::item_type<db::add_tag_prefix<BoundaryContribution, variables_tag>>
        result_in_volume{num_points, 0.};
    boundary_scheme::apply(
        make_not_null(&result_in_volume), make_not_null(&all_mortar_data), mesh,
        {{mortar_id, mortar_mesh}}, {{mortar_id, mortar_size}}, numerical_flux);
    auto result_on_face =
        extract_on_face(make_not_null(&result_in_volume), mesh, slice_dim);
    // Projected F* - F = {5., -1.}
    Variables<tmpl::list<Tags::NormalDotNumericalFlux<SomeField>>>
        fstar_minus_f{face_num_points};
    get(get<Tags::NormalDotNumericalFlux<SomeField>>(fstar_minus_f)) =
        DataVector{5., -1.};
    const auto expected =
        dg::lift_flux(fstar_minus_f, extent_perpendicular_to_boundary,
                      magnitude_of_face_normal);
    CHECK_ITERABLE_APPROX(get(result_on_face), get(get<SomeField>(expected)));
  }
  {
    // This test was carried over from Test_MortarHelpers.cpp
    INFO("h-refinement");
    constexpr size_t Dim = 2;
    using boundary_scheme = dg::BoundarySchemes::StrongFirstOrder<
        Dim, variables_tag, NumericalFluxTag<RefinementTestsNumericalFlux>,
        TemporalIdTag>;
    const Mesh<Dim> mesh{{{4, 3}},
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    const auto compute_contribution = [&mesh](
        const MortarId<Dim>& mortar_id,
        const std::array<Spectral::MortarSize, Dim - 1>& mortar_size,
        const DataVector& numerical_flux) noexcept {
      const auto mortar_mesh = mesh.slice_away(mortar_id.first.dimension());
      const size_t mortar_num_points = mortar_mesh.number_of_grid_points();

      // These are all arbitrary
      const DataVector local_flux{-1., 5., 7.};
      const DataVector magnitude_of_face_normal{2., 5., 7.};

      typename boundary_scheme::BoundaryData interior_data{mortar_num_points};
      get(get<Tags::NormalDotFlux<SomeField>>(interior_data.field_data)) =
          local_flux;
      get(get<::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim>>>(
          interior_data.extra_data)) = magnitude_of_face_normal;
      get<SomeField>(interior_data.field_data) =
          Scalar<DataVector>{mortar_num_points, 0.};
      interior_data = interior_data.project_to_mortar(mortar_mesh, mortar_mesh,
                                                      mortar_size);
      get(get<SomeField>(interior_data.field_data)) = DataVector{1., 2., 3.};

      typename boundary_scheme::BoundaryData exterior_data{mortar_num_points};
      get(get<SomeField>(exterior_data.field_data)) = DataVector{6., 5., 4.};

      auto all_mortar_data = make_mortar_data<boundary_scheme>(
          mortar_id, 0, std::move(interior_data), std::move(exterior_data));

      db::item_type<db::add_tag_prefix<BoundaryContribution, variables_tag>>
          result_in_volume{mesh.number_of_grid_points(), 0.};
      boundary_scheme::apply(
          make_not_null(&result_in_volume), make_not_null(&all_mortar_data),
          mesh, {{mortar_id, mortar_mesh}}, {{mortar_id, mortar_size}},
          RefinementTestsNumericalFlux{numerical_flux});
      return result_in_volume;
    };

    const auto unrefined_result =
        compute_contribution({Direction<Dim>::upper_xi(), ElementId<Dim>{0}},
                             {{Spectral::MortarSize::Full}}, {1., 4., 9.});
    const decltype(unrefined_result) refined_result =
        compute_contribution({Direction<Dim>::upper_xi(), ElementId<Dim>{0}},
                             {{Spectral::MortarSize::LowerHalf}},
                             {1., 9. / 4., 4.}) +
        compute_contribution({Direction<Dim>::upper_xi(), ElementId<Dim>{1}},
                             {{Spectral::MortarSize::UpperHalf}},
                             {4., 25. / 4., 9.});
    CHECK_VARIABLES_APPROX(unrefined_result, refined_result);
  }
}
