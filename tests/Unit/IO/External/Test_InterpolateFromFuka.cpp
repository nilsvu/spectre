// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/External/InterpolateFromFuka.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.IO.External.InterpolateFromFuka", "[Unit][IO]") {
  std::mutex fuka_lock{};
  // Get example data directory from environment variable
  const char* fuka_root_ptr = std::getenv("FUKA_ROOT");
  REQUIRE(fuka_root_ptr != nullptr);
  const std::string fuka_root{fuka_root_ptr};
  REQUIRE_FALSE(fuka_root.empty());
  CAPTURE(fuka_root);
  const std::string example_id_dir = fuka_root + "/example_id";
  {
    INFO("BH");
    const tnsr::I<DataVector, 3> coords{{{{2.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bh>(
        make_not_null(&fuka_lock),
        example_id_dir + "/converged_BH_TOTAL_BC.0.5.0.0.09.info", coords);
    CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataVector>>(fuka_data)),
                          DataVector{0.30512109920956748});
  }
  {
    INFO("BBH");
    const tnsr::I<DataVector, 3> coords{{{{0.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bbh>(
        make_not_null(&fuka_lock),
        example_id_dir + "/converged_BBH_TOTAL_BC.10.0.0.1.q1.0.0.09.info",
        coords);
    CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataVector>>(fuka_data)),
                          DataVector{0.0000042161179477});
  }
  {
    INFO("NS");
    const tnsr::I<DataVector, 3> coords{{{{0.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Ns>(
        make_not_null(&fuka_lock),
        example_id_dir + "/converged_NS_TOTAL_BC.togashi.2.23.-0.4.0.11.info",
        coords);
    CHECK_ITERABLE_APPROX(
        get(get<hydro::Tags::RestMassDensity<DataVector>>(fuka_data)),
        DataVector{0.00220590213673744});
  }
  {
    INFO("BNS");
    const tnsr::I<DataVector, 3> coords{{{{15.3}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bns>(
        make_not_null(&fuka_lock),
        example_id_dir +
            "/converged_BNS_TOTAL.togashi.30.6.0.0.2.8.q1.0.0.09.info",
        coords);
    CHECK_ITERABLE_APPROX(
        get(get<hydro::Tags::RestMassDensity<DataVector>>(fuka_data)),
        DataVector{0.00093178076659427});
  }
  {
    INFO("BHNS");
    const tnsr::I<DataVector, 3> coords{{{{0.0}, {0.0}, {0.0}}}};
    const auto fuka_data = io::interpolate_from_fuka<io::FukaIdType::Bhns>(
        make_not_null(&fuka_lock),
        example_id_dir +
            "/converged_BHNS_ECC_RED.togashi.35.0.6.0.52.3.6.q0.487603.0.1.11."
            "info",
        coords);
    CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataVector>>(fuka_data)),
                          DataVector{0.08808942723720847});
  }
}
