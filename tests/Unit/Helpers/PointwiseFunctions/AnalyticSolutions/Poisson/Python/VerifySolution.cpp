// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PythonBindings/BoostOptional.hpp"
#include "Utilities/GetOutput.hpp"

namespace py = pybind11;

namespace TestHelpers {
namespace Poisson {
namespace Solutions {
namespace py_bindings {

namespace {

template <size_t Dim, typename SolutionType>
auto verify_solution_impl(
    const SolutionType& solution, const DomainCreator<Dim>& domain_creator,
    const boost::optional<std::string>& dump_to_file = boost::none) {
  using system =
      ::Poisson::FirstOrderSystem<Dim, ::Poisson::Geometry::Euclidean>;
  const typename system::fluxes fluxes_computer{};
  // Apply the first-order DG operator to the solution. This computes the
  // residual `Ax - b` where `A` is the DG operator, `x` is the solution
  // evaluated on the `domain_creator`'s grid and `b` is the fixed source given
  // by the solution.
  const auto dg_operator_applied_to_solution =
      TestHelpers::elliptic::dg::apply_dg_operator_to_solution<system>(
          solution, domain_creator,
          [&fluxes_computer](const auto&... args) {
            return TestHelpers::elliptic::dg::apply_first_order_dg_operator<
                system>(
                args..., fluxes_computer,
                // The fluxes and sources need no arguments, so we return
                // empty tuples
                [](const auto&... /* unused */) { return std::tuple<>{}; },
                [](const auto&... /* unused */) { return std::tuple<>{}; },
                // Disable boundary terms since the analytic solutions are
                // continuous across element boundaries.
                [](const auto&... /* unused */) {
                  return TestHelpers::elliptic::dg::EmptyBoundaryData{};
                },
                [](const auto&... /* unused */) {});
          },
          dump_to_file);
  // Convert the data to a type that Python understands. We just take a
  // norm of the Poisson field over each element for now.
  std::unordered_map<ElementId<Dim>, double> result{};
  for (const auto& id_and_var : dg_operator_applied_to_solution) {
    const auto& element_id = id_and_var.first;
    result[element_id] =
        l2_norm(get<::Poisson::Tags::Field>(id_and_var.second));
  }
  return result;
}

template <size_t Dim>
void bind_verify_product_of_sinusoids(py::module& m) {  // NOLINT
  m.def(
      ("verify_product_of_sinusoids_" + get_output(Dim) + "d").c_str(),
      &verify_solution_impl<Dim, ::Poisson::Solutions::ProductOfSinusoids<Dim>>,
      py::arg("solution"), py::arg("domain_creator"),
      py::arg("dump_to_file") = boost::optional<std::string>{boost::none});
}
}  // namespace

void bind_verify_solution(py::module& m) {  // NOLINT
  bind_verify_product_of_sinusoids<1>(m);
  bind_verify_product_of_sinusoids<2>(m);
  bind_verify_product_of_sinusoids<3>(m);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Poisson
}  // namespace TestHelpers
