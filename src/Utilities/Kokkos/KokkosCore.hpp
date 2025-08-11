// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#if __has_include(<Kokkos_Core.hpp>)
#include <Kokkos_Core.hpp>

/// \brief If defined then SpECTRE is using Kokkos
#define SPECTRE_KOKKOS 1

using MatrixViewRO =
    Kokkos::View<const double**, Kokkos::LayoutLeft,
                 typename Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

#else  // #if __has_include(<Kokkos_Core.hpp>)
#define KOKKOS_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#endif  // #if __has_include(<Kokkos_Core.hpp>)
