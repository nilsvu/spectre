// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

template <typename T>
struct AtIndex {};

template <typename... Ts, typename Symm, template <typename...> class IndexList,
          typename... Indices>
class Tensor<AtIndex<Kokkos::View<Ts...>>, Symm, IndexList<Indices...>> {
 public:
  using storage_type =
      Kokkos::View<Ts...>[Tensor_detail::Structure<Symm, Indices...>::size()];
  // The type of the stored grid point indices
  using indices_type = int[Kokkos::View<Ts...>::rank()];
  using type = Kokkos::View<Ts...>;
  using symmetry = Symm;
  using index_list = tmpl::list<Indices...>;
  static constexpr size_t num_tensor_indices = sizeof...(Indices);
  using structure = Tensor_detail::Structure<Symm, Indices...>;

  Tensor() = default;
  template <typename... Ints>
  KOKKOS_FUNCTION Tensor(const Tensor<Kokkos::View<Ts...>, Symm,
                                      IndexList<Indices...>>& tensor_view,
                         Ints... ints)
      : indices_{static_cast<int>(ints)...} {
    static_assert(sizeof...(Ints) == Kokkos::View<Ts...>::rank());
    for (int i = 0; i < static_cast<int>(structure::size()); ++i) {
      data_[i] = tensor_view[static_cast<size_t>(i)];
    }
  }

  using value_type = typename type::value_type;
  using reference = typename type::reference_type;
  using const_reference = typename type::reference_type;
  using pointer = typename type::pointer_type;
  using const_pointer = typename type::pointer_type;

  template <typename... N>
  constexpr reference get(N... n) {
    static_assert(
        sizeof...(Indices) == sizeof...(N),
        "the number of tensor indices specified must match the rank of "
        "the tensor");
    if constexpr (Kokkos::View<Ts...>::rank() == 1) {
      return data_[structure::get_storage_index(n...)](indices_[0]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 2) {
      return data_[structure::get_storage_index(n...)](indices_[0],
                                                       indices_[1]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 3) {
      return data_[structure::get_storage_index(n...)](indices_[0], indices_[1],
                                                       indices_[2]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 4) {
      return data_[structure::get_storage_index(n...)](
          indices_[0], indices_[1], indices_[2], indices_[3]);
    }
  }

  template <typename... N>
  constexpr const_reference get(N... n) const {
    static_assert(
        sizeof...(Indices) == sizeof...(N),
        "the number of tensor indices specified must match the rank of "
        "the tensor");
    if constexpr (Kokkos::View<Ts...>::rank() == 1) {
      return data_[structure::get_storage_index(n...)](indices_[0]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 2) {
      return data_[structure::get_storage_index(n...)](indices_[0],
                                                       indices_[1]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 3) {
      return data_[structure::get_storage_index(n...)](indices_[0], indices_[1],
                                                       indices_[2]);
    } else if constexpr (Kokkos::View<Ts...>::rank() == 4) {
      return data_[structure::get_storage_index(n...)](
          indices_[0], indices_[1], indices_[2], indices_[3]);
    }
  }

  template <int... N, typename... Us, typename... Args>
  friend constexpr
      typename Tensor<AtIndex<Kokkos::View<Us...>>, Args...>::reference
      get(Tensor<AtIndex<Kokkos::View<Us...>>, Args...>& t);
  template <int... N, typename... Us, typename... Args>
  friend constexpr
      typename Tensor<AtIndex<Kokkos::View<Us...>>, Args...>::const_reference
      get(const Tensor<AtIndex<Kokkos::View<Us...>>, Args...>& t);

  SPECTRE_ALWAYS_INLINE static constexpr size_t size() {
    return structure::size();
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t rank() {
    return sizeof...(Indices);
  }

 private:
  storage_type data_;
  indices_type indices_{};
};

template <int... N, typename... Us, typename... Args>
constexpr typename Tensor<AtIndex<Kokkos::View<Us...>>, Args...>::reference get(
    Tensor<AtIndex<Kokkos::View<Us...>>, Args...>& t) {
  static_assert(
      Tensor<AtIndex<Kokkos::View<Us...>>, Args...>::rank() == sizeof...(N),
      "the number of tensor indices specified must match the rank "
      "of the tensor");
  return t.get(N...);
}

template <int... N, typename... Us, typename... Args>
constexpr
    typename Tensor<AtIndex<Kokkos::View<Us...>>, Args...>::const_reference
    get(const Tensor<AtIndex<Kokkos::View<Us...>>, Args...>& t) {
  static_assert(
      Tensor<AtIndex<Kokkos::View<Us...>>, Args...>::rank() == sizeof...(N),
      "the number of tensor indices specified must match the rank "
      "of the tensor");
  return t.get(N...);
}
