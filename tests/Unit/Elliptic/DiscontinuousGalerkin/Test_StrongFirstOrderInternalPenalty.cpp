// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/InternalPenalty.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "tests/Unit/Elliptic/DiscontinuousGalerkin/OperatorMatrixTestHelpers.hpp"

#include "Domain/MirrorVariables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/PopulateBoundaryMortars.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"

#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarFieldTag"; }
};

template <size_t Dim>
using FieldsTag = Tags::Variables<
    tmpl::list<ScalarFieldTag, ::Tags::deriv<ScalarFieldTag, tmpl::size_t<Dim>,
                                             Frame::Inertial>>>;

template <size_t Dim>
struct CharacteristicSpeedsCompute : db::ComputeTag {
  static std::string name() noexcept { return "CharacteristicSpeeds"; }

  using argument_tags = tmpl::list<::Tags::Mesh<Dim - 1>>;

  using return_type = std::array<DataVector, 1 + Dim>;
  static void function(gsl::not_null<return_type*> result,
                       const Mesh<Dim - 1>& mesh) noexcept {
    const size_t num_points = mesh.number_of_grid_points();
    for (size_t i = 0; i < 1 + Dim; i++) {
      gsl::at(*result, i) = DataVector{num_points, -2.};
    }
  }
};

template <size_t Dim>
struct System {
  using variables_tag = FieldsTag<Dim>;
  using char_speeds_tag = CharacteristicSpeedsCompute<Dim>;
};

template <size_t Dim>
using strong_first_order_internal_penalty_scheme =
    dg::BoundarySchemes::StrongFirstOrder<
        Dim, FieldsTag<Dim>,
        ::Tags::NormalDotNumericalFluxComputer<
            dg::NumericalFluxes::FirstOrderInternalPenalty<
                Dim, tmpl::list<ScalarFieldTag>,
                tmpl::list<::Tags::deriv<ScalarFieldTag, tmpl::size_t<Dim>,
                                         Frame::Inertial>>>>,
        OperatorMatrixTestHelpers::TemporalIdTag,
        Poisson::ComputeFirstOrderFluxes<
            Dim, FieldsTag<Dim>, ScalarFieldTag,
            ::Tags::deriv<ScalarFieldTag, tmpl::size_t<Dim>, Frame::Inertial>>,
        Poisson::ComputeFirstOrderSources<
            Dim, FieldsTag<Dim>, ScalarFieldTag,
            ::Tags::deriv<ScalarFieldTag, tmpl::size_t<Dim>, Frame::Inertial>>>;

template <size_t Dim>
using extra_initialization_actions =
    tmpl::list<Initialization::Actions::AddComputeTags<tmpl::list<
        ::Tags::InterfaceComputeItem<
            ::Tags::BoundaryDirectionsExterior<Dim>,
            ::Tags::MirrorVariables<
                Dim, ::Tags::BoundaryDirectionsInterior<Dim>, FieldsTag<Dim>,
                tmpl::list<ScalarFieldTag>>>,

        ::Tags::Slice<
            ::Tags::InternalDirections<Dim>,
            db::add_tag_prefix<
                ::Tags::div,
                db::add_tag_prefix<::Tags::Flux, FieldsTag<Dim>,
                                   tmpl::size_t<Dim>, Frame::Inertial>>>,
        ::Tags::InterfaceComputeItem<
            ::Tags::InternalDirections<Dim>,
            Poisson::ComputeFirstOrderFluxes<
                Dim,
                db::add_tag_prefix<
                    ::Tags::div,
                    db::add_tag_prefix<::Tags::Flux, FieldsTag<Dim>,
                                       tmpl::size_t<Dim>, Frame::Inertial>>,
                ScalarFieldTag,
                ::Tags::div<::Tags::Flux<
                    ::Tags::deriv<ScalarFieldTag, tmpl::size_t<Dim>,
                                  Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>>>>,
        ::Tags::InterfaceComputeItem<
            ::Tags::InternalDirections<Dim>,
            Poisson::ComputeFirstOrderNormalFluxes<
                Dim, FieldsTag<Dim>, ScalarFieldTag,
                ::Tags::deriv<ScalarFieldTag, tmpl::size_t<Dim>,
                              Frame::Inertial>>>,

        ::Tags::InterfaceComputeItem<::Tags::InternalDirections<Dim>,
                                     CharacteristicSpeedsCompute<Dim>>,
        ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsInterior<Dim>,
                                     CharacteristicSpeedsCompute<Dim>>,
        ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<Dim>,
                                     CharacteristicSpeedsCompute<Dim>>>>>;

template <typename DgScheme>
using extra_iterable_actions =
    tmpl::list<Actions::MutateApply<dg::PopulateBoundaryMortars<DgScheme>>>;

}  // namespace

// [[TimeOut, 60]]
SPECTRE_TEST_CASE(
    "Unit.Elliptic.StrongFirstOrderInternalPenalty.OperatorMatrix",
    "[Elliptic][Unit]") {
  // These tests build the matrix representation of the DG operator and compare
  // it to a matrix that was computed independently using the code available at
  // https://github.com/nilsleiffischer/dgpy at commit c0a87ce.

  domain::creators::register_derived_with_charm();
  {
    INFO("1D");
    using dg_scheme = strong_first_order_internal_penalty_scheme<1>;
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{0.}}, {{M_PI}}, {{false}}, {{1}}, {{3}}};
    dg::NumericalFluxes::LocalLaxFriedrichs<System<1>> numflux{};
    OperatorMatrixTestHelpers::test_operator_matrix<
        dg_scheme, extra_initialization_actions<1>,
        extra_iterable_actions<dg_scheme>, ScalarFieldTag>(
        std::move(domain_creator), "DgFirstOrderPoissonOperator1DSample.dat",
        numflux);
  }
}
