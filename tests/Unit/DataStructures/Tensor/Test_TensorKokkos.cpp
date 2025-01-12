// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {

template <typename DataType>
KOKKOS_FUNCTION void tensor_func(gsl::not_null<Scalar<DataType>*> result,
                                 const Scalar<DataType>& input) {
  get(*result) = 2.0 * get(input);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Kokkos",
                  "[Unit][DataStructures]") {
  {
    // Allocate device memory for tensor components. This can be replaced with a
    // Kokkos specialization of Variables.
    const size_t num_components = 1;
    const size_t num_points = 3;
    Kokkos::View<double**> vars{"vars", num_components, num_points};
    Kokkos::parallel_for(
        "fill",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>{{0, 0},
                                               {num_components, num_points}},
        KOKKOS_LAMBDA(const int component, const int i) {
          vars(component, i) = 2.0 * i;
        });

    // Invoke pointwise tensor operations
    Kokkos::parallel_for(
        "compute", num_points, KOKKOS_LAMBDA(const int i) {
          Scalar<std::reference_wrapper<double>> scalar_at_index{vars(0, i)};
          tensor_func(make_not_null(&scalar_at_index), scalar_at_index);
        });

    // Check result
    Kokkos::fence();
    const auto vars_host = Kokkos::create_mirror_view(vars);
    Kokkos::deep_copy(vars_host, vars);
    for (int i = 0; i < num_points; ++i) {
      CHECK(vars_host(0, i) == 4.0 * i);
    }
  }
}
