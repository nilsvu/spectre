// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdlib>
#include <memory>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/FukaInitialData.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.FukaInitialData",
                  "[Unit][PointwiseFunctions]") {
  register_classes_with_charm<grmhd::AnalyticData::FukaInitialData>();

  // Get example data directory from environment variable
  const char* fuka_root_ptr = std::getenv("FUKA_ROOT");
  REQUIRE(fuka_root_ptr != nullptr);
  const std::string fuka_root{fuka_root_ptr};
  REQUIRE_FALSE(fuka_root.empty());
  CAPTURE(fuka_root);
  const std::string example_id_dir = fuka_root + "/example_id";

  const auto option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::FukaInitialData>(
          "FukaInitialData:\n"
          "  InfoFilename: \"" +
          example_id_dir +
          "/converged_BNS_TOTAL.togashi.30.6.0.0.2.8.q1.0.0.09.info\"\n"
          "  ElectronFraction: 0.15")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& solution =
      dynamic_cast<const grmhd::AnalyticData::FukaInitialData&>(
          *deserialized_option_solution);

  const tnsr::I<DataVector, 3> coords{{{{15.3}, {0.0}, {0.0}}}};
  const auto fuka_data =
      solution.variables(coords, FukaInitialData::tags<DataVector>{});

  CHECK_ITERABLE_APPROX(
      get(get<hydro::Tags::RestMassDensity<DataVector>>(fuka_data)),
      DataVector{0.00093178076659427});
}

}  // namespace grmhd::AnalyticData
