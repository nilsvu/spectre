// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "IO/DataImporter/Tags.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

namespace {
struct ExampleDataGroup {
  using group = importer::OptionTags::Group;
  static std::string name() noexcept { return "ExampleData"; }
  static constexpr OptionString help = "halp";
};
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.DataImporter.Tags", "[Unit][IO]") {
  CHECK(db::tag_name<importer::Tags::RegisteredElements>() ==
        "RegisteredElements");
  CHECK(db::tag_name<importer::Tags::DataFileName<ExampleDataGroup>>() ==
        "DataFileName(ExampleData)");
  CHECK(db::tag_name<importer::Tags::VolumeDataSubgroup<ExampleDataGroup>>() ==
        "VolumeDataSubgroup(ExampleData)");
  CHECK(db::tag_name<importer::Tags::ObservationValue<ExampleDataGroup>>() ==
        "ObservationValue(ExampleData)");

  Options<tmpl::list<importer::OptionTags::DataFileName<ExampleDataGroup>,
                     importer::OptionTags::VolumeDataSubgroup<ExampleDataGroup>,
                     importer::OptionTags::ObservationValue<ExampleDataGroup>>>
      opts("");
  opts.parse(
      "DataImporters:\n"
      "  ExampleData:\n"
      "    DataFileName: File.name\n"
      "    VolumeDataSubgroup: data.group\n"
      "    ObservationValue: 1.");
  CHECK(opts.get<importer::OptionTags::DataFileName<ExampleDataGroup>>() ==
        "File.name");
  CHECK(
      opts.get<importer::OptionTags::VolumeDataSubgroup<ExampleDataGroup>>() ==
      "data.group");
  CHECK(opts.get<importer::OptionTags::ObservationValue<ExampleDataGroup>>() ==
        1.);
}
