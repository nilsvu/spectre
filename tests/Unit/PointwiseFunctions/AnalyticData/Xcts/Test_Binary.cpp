// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/FirstOrderEllipticSolutionsTestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::AnalyticData {
namespace {

void test_data() {
  using isolated_object_registrars =
      tmpl::list<Xcts::Solutions::Registrars::Kerr>;
  const auto created = TestHelpers::test_factory_creation<
      AnalyticData<tmpl::list<Registrars::Binary<isolated_object_registrars>>>>(
      "Binary:\n"
      "  XCoords: [-5., 100.]\n"
      "  ObjectA:\n"
      "    Kerr:\n"
      "      Center: [0., 0., 0.]\n"
      "      Spin: [0., 0., 0.]\n"
      "      Mass: 0.43\n"
      "  ObjectB:\n"
      "    Kerr:\n"
      "      Center: [0., 0., 0.]\n"
      "      Spin: [0., 0., 0.]\n"
      "      Mass: 1.\n"
      "  AngularVelocity: 0.02\n"
      "  FalloffWidths: [100., 8.]");
  REQUIRE(dynamic_cast<const Binary<isolated_object_registrars>*>(
              created.get()) != nullptr);
  const auto& binary =
      dynamic_cast<const Binary<isolated_object_registrars>&>(*created);
  {
    INFO("Properties");
    CHECK(binary.angular_velocity() == 0.02);
  }
  {
    tnsr::I<DataVector, 3> x_inner_left{size_t{1}, 0.};
    get<0>(x_inner_left) = 100. + 0.86;
    const auto vars_inner_left = binary.variables<>(
        x_inner_left,
        tmpl::list<Tags::LapseTimesConformalFactor<DataVector>>{});
    CAPTURE(vars_inner_left);
    CHECK(false);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.Xcts.Binary",
                  "[PointwiseFunctions][Unit]") {
  test_data();
}

}  // namespace Xcts::AnalyticData
