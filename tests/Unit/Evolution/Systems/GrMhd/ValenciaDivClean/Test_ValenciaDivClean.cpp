// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.ValenciaDivClean.Tags",
                  "[Unit][GrMhd]") {
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeeds>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<grmhd::ValenciaDivClean::Tags::TildeD>(
      "TildeD");
  TestHelpers::db::test_simple_tag<grmhd::ValenciaDivClean::Tags::TildeYe>(
      "TildeYe");
  TestHelpers::db::test_simple_tag<grmhd::ValenciaDivClean::Tags::TildeTau>(
      "TildeTau");
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::TildeS<Frame::Grid>>("Grid_TildeS");
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::TildeB<Frame::Grid>>("Grid_TildeB");
  TestHelpers::db::test_simple_tag<grmhd::ValenciaDivClean::Tags::TildePhi>(
      "TildePhi");
  TestHelpers::db::test_simple_tag<
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter>(
      "ConstraintDampingParameter");
}
