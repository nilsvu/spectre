// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/optional.hpp>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/Systems/Xcts/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/ConstantDensityStar.hpp"
#include "Utilities/GetOutput.hpp"

// Allow using boost::optional in Python bindings
namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};
}  // namespace detail
}  // namespace pybind11

namespace py = pybind11;

namespace TestHelpers {
namespace Xcts {
namespace Solutions {
namespace py_bindings {

namespace {

template <typename SolutionType>
auto verify_solution_impl(
    const SolutionType& solution, const DomainCreator<3>& domain_creator,
    const boost::optional<std::string>& dump_to_file = boost::none) {
  using system = ::Xcts::FirstOrderSystem<::Xcts::Equations::Hamiltonian>;
  const typename system::fluxes fluxes_computer{};
  const auto dg_operator_applied_to_solution =
      TestHelpers::elliptic::dg::apply_dg_operator_to_solution<system>(
          solution, domain_creator,
          [&solution, &fluxes_computer](const auto&... args) {
            return TestHelpers::elliptic::dg::apply_first_order_dg_operator<
                system>(
                args..., fluxes_computer,
                [](const auto&... /* unused */) { return std::tuple<>{}; },
                [&solution](const auto& /* element_id */,
                            const auto& dg_element) {
                  const auto logical_coords =
                      logical_coordinates(dg_element.mesh);
                  const auto inertial_coords =
                      dg_element.element_map(logical_coords);
                  return std::make_tuple(get<
                                         gr::Tags::EnergyDensity<DataVector>>(
                      solution.variables(
                          inertial_coords,
                          tmpl::list<gr::Tags::EnergyDensity<DataVector>>{})));
                },
                [](const auto&... /* unused */) {
                  return TestHelpers::elliptic::dg::EmptyBoundaryData{};
                },
                [](const auto&... /* unused */) {});
          },
          dump_to_file);
  // Convert the data to a type that Python understands. We just take a
  // norm of the Poisson field over each element for now.
  std::unordered_map<ElementId<3>, std::pair<double, double>> result{};
  for (const auto& id_and_var : dg_operator_applied_to_solution) {
    const auto& element_id = id_and_var.first;
    result[element_id] = std::make_pair(
        l2_norm(
            get<::Xcts::Tags::ConformalFactor<DataVector>>(id_and_var.second)),
        l2_norm(get<::Tags::deriv<::Xcts::Tags::ConformalFactor<DataVector>,
                                  tmpl::size_t<3>, Frame::Inertial>>(
            id_and_var.second)));
  }
  return result;
}

void bind_verify_constant_density_star(py::module& m) {  // NOLINT
  m.def("verify_constant_density_star",
        &verify_solution_impl<::Xcts::Solutions::ConstantDensityStar>,
        py::arg("solution"), py::arg("domain_creator"),
        py::arg("dump_to_file") = boost::optional<std::string>{boost::none});
}
}  // namespace

void bind_verify_solution(py::module& m) {  // NOLINT
  bind_verify_constant_density_star(m);
}

}  // namespace py_bindings
}  // namespace Solutions
}  // namespace Xcts
}  // namespace TestHelpers
