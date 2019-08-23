// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <blaze/math/Serialization.h>
#include <blaze/util/serialization/Archive.h>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Helpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication2.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeInterfaces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"

namespace OperatorMatrixTestHelpers {

template <typename Tag>
struct DgOperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "DgOperatorAppliedTo(" + db::tag_name<Tag>() + ")";
  }
};

struct TemporalIdTag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "TemporalIdTag"; }
  template <typename Tag>
  using step_prefix = DgOperatorAppliedTo<Tag>;
};

struct AdvanceTemporalId {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using temporal_id_tag = TemporalIdTag;
    db::mutate<temporal_id_tag, Tags::Next<temporal_id_tag>>(
        make_not_null(&box), [](const gsl::not_null<int*> temporal_id,
                                const gsl::not_null<int*> next_temporal_id) {
          *temporal_id = *next_temporal_id;
          (*next_temporal_id)++;
        });
    return std::move(box);
  }
};

template <size_t Dim, typename Metavariables,
          typename ExtraInitializationActions, typename ExtraIterableActions>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;

  using dg_scheme = typename Metavariables::dg_scheme;
  using temporal_id_tag = typename dg_scheme::temporal_id_tag;
  using variables_tag = typename dg_scheme::variables_tag;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::flatten<tmpl::list<
              ActionTesting::InitializeDataBox<
                  tmpl::list<::Tags::Domain<Dim, Frame::Inertial>,
                             ::Tags::InitialExtents<Dim>, temporal_id_tag,
                             ::Tags::Next<temporal_id_tag>, variables_tag>>,
              dg::Actions::InitializeDomain<Dim>,
              dg::Actions::InitializeInterfaces<
                  typename Metavariables::system,
                  dg::Initialization::slice_tags_to_face<variables_tag>,
                  dg::Initialization::slice_tags_to_exterior<>>,
              dg::Actions::InitializeMortars<
                  dg_scheme, dg_scheme::use_external_mortars,
                  Initialization::MergePolicy::Overwrite>,
              ExtraInitializationActions,
              typename dg_scheme::initialize_element>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::flatten<
              tmpl::list<dg::Actions::SendDataForFluxes<dg_scheme>,
                         dg::Actions::ReceiveDataForFluxes<dg_scheme>,
                         ExtraIterableActions, Actions::MutateApply<dg_scheme>,
                         AdvanceTemporalId>>>>;

  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags_from_pdal<
          phase_dependent_action_list>;
};

// Only needed to provide magnitude to InitializeInterfaces
template <size_t Dim, typename VariablesTag>
struct System {
  static constexpr size_t volume_dim = Dim;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <typename DgScheme, typename ExtraInitializationActions,
          typename ExtraIterableActions>
struct Metavariables {
  using dg_scheme = DgScheme;
  static constexpr size_t volume_dim = dg_scheme::volume_dim;
  using system = System<volume_dim, typename dg_scheme::variables_tag>;
  using temporal_id_tag = TemporalIdTag;
  using element_array =
      ElementArray<volume_dim, Metavariables, ExtraInitializationActions,
                   ExtraIterableActions>;
  using component_list = tmpl::list<element_array>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename VariablesTag>
struct SetData {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, size_t Dim,
            Requires<db::tag_is_retrievable_v<VariablesTag, DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& /*array_index*/,
                    const db::item_type<VariablesTag>& variables) noexcept {
    db::mutate<VariablesTag>(
        make_not_null(&box), [&variables](const gsl::not_null<
                                          db::item_type<VariablesTag>*>
                                              local_variables) noexcept {
          *local_variables = variables;
        });
  }
};

// Only works for a particular scalar field right now, generalize to all fields
// in the vars
template <typename DgScheme, typename ExtraInitializationActions,
          typename ExtraIterableActions, typename ScalarFieldTag,
          typename... CacheArgs, size_t Dim = DgScheme::volume_dim>
void test_operator_matrix(
    const DomainCreator<Dim, Frame::Inertial>& domain_creator,
    const std::string& expected_matrix_filename,
    const CacheArgs&... cache_args) {
  using metavariables =
      Metavariables<DgScheme, ExtraInitializationActions, ExtraIterableActions>;
  using element_array = typename metavariables::element_array;

  using dg_scheme = DgScheme;
  using variables_tag = typename dg_scheme::variables_tag;
  using temporal_id_tag = typename dg_scheme::temporal_id_tag;
  using operator_applied_to_variables_tag =
      db::add_tag_prefix<temporal_id_tag::template step_prefix, variables_tag>;

  const auto domain = domain_creator.create_domain();
  const auto initial_extents = domain_creator.initial_extents();
  const auto& blocks = domain.blocks();
  std::vector<ElementId<Dim>> element_ids{};
  for (const auto& block : blocks) {
    const auto initial_ref_levs =
        domain_creator.initial_refinement_levels()[block.id()];
    const auto block_element_ids =
        initial_element_ids(block.id(), initial_ref_levs);
    element_ids.insert(element_ids.end(), block_element_ids.begin(),
                       block_element_ids.end());
  }

  ActionTesting::MockRuntimeSystem<metavariables> runner{{cache_args...}};

  size_t operator_size = 0;

  // Set up all elements in the domain
  for (const auto& element_id : element_ids) {
    // The variables are set later for testing different values
    const size_t num_points =
        ::Initialization::element_mesh(initial_extents, element_id)
            .number_of_grid_points();
    db::item_type<variables_tag> vars{num_points};
    operator_size += vars.size();

    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents(), 0, 1,
         std::move(vars)});
    for (size_t i_init_actions = 0;
         i_init_actions < 4 + tmpl::size<ExtraInitializationActions>::value;
         i_init_actions++) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
  }

  runner.set_phase(metavariables::Phase::Testing);

  DenseMatrix<double, blaze::rowMajor> dg_operator_matrix{operator_size,
                                                          operator_size};

  // Build the matrix by applying the operator to unit vectors
  size_t i_across_elements = 0;
  size_t j_across_elements = 0;
  for (const auto& active_element : element_ids) {
    const size_t size_active_element =
        ActionTesting::get_databox_tag<element_array, variables_tag>(
            runner, active_element)
            .size();
    for (size_t i = 0; i < size_active_element; i++) {
      for (const auto& element_id : element_ids) {
        const auto& mesh =
            ActionTesting::get_databox_tag<element_array, ::Tags::Mesh<Dim>>(
                runner, element_id);

        // Construct a unit vector
        db::item_type<variables_tag> vars{mesh.number_of_grid_points(), 0.};
        if (element_id == active_element) {
          vars.data()[i] = 1.;
        }
        ActionTesting::simple_action<element_array, SetData<variables_tag>>(
            make_not_null(&runner), element_id, std::move(vars));

        // Send data
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
      }
      // Split the loop to have all elements send their data before receiving
      for (const auto& element_id : element_ids) {
        // Receive data
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
        // Run extra actions
        for (size_t i_extra_actions = 0;
             i_extra_actions < tmpl::size<ExtraIterableActions>::value;
             i_extra_actions++) {
          ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                    element_id);
        }

        // Apply the operator
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
        const auto& operator_applied_to_vars =
            ActionTesting::get_databox_tag<element_array,
                                           operator_applied_to_variables_tag>(
                runner, element_id);

        // Store result in matrix
        for (size_t j = 0; j < operator_applied_to_vars.size(); j++) {
          dg_operator_matrix(i_across_elements, j_across_elements) =
              operator_applied_to_vars.data()[j];
          j_across_elements++;
        }

        // Advance temporal ID
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
      }
      i_across_elements++;
      j_across_elements = 0;
    }
  }

  //   Parallel::printf("%s\n", dg_operator_matrix);
  const std::string outpath = unit_test_path() +
                              "/Elliptic/DiscontinuousGalerkin/" + +"out_" +
                              expected_matrix_filename;
  if (file_system::check_if_file_exists(outpath)) {
    file_system::rm(outpath, true);
  }
  std::ofstream out_matrix_file{outpath};
  for (size_t row = 0; row < operator_size; row++) {
    for (DenseMatrix<double, blaze::rowMajor>::Iterator matrix_element =
             dg_operator_matrix.begin(row);
         matrix_element != dg_operator_matrix.end(row); matrix_element++) {
      out_matrix_file << std::setprecision(18) << *matrix_element << " ";
    }
    out_matrix_file << "\n";
  }

  // Load expected matrix from file
  // std::ifstream expected_matrix_file{unit_test_path() +
  //                                    "/Elliptic/DiscontinuousGalerkin/" +
  //                                    expected_matrix_filename};
  // std::istream_iterator<double> expected_matrix_file_element{
  //     expected_matrix_file};
  // DenseMatrix<double, blaze::rowMajor> expected_matrix{operator_size,
  //                                                      operator_size};
  // for (size_t row = 0; row < operator_size; row++) {
  //   for (DenseMatrix<double, blaze::rowMajor>::Iterator
  //            expected_matrix_element = expected_matrix.begin(row);
  //        expected_matrix_element != expected_matrix.end(row);
  //        expected_matrix_element++) {
  //     *expected_matrix_element = *expected_matrix_file_element;
  //     expected_matrix_file_element++;
  //   }
  // }
  // // Make sure we have reached the end of the file
  // CHECK(expected_matrix_file_element == std::istream_iterator<double>{});

  // CHECK_MATRIX_APPROX(dg_operator_matrix, expected_matrix);
}

}  // namespace OperatorMatrixTestHelpers
