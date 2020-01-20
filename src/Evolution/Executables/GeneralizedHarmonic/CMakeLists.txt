# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    ApparentHorizons
    CoordinateMaps
    DiscontinuousGalerkin
    Domain
    DomainCreators
    GeneralRelativity
    GeneralizedHarmonic
    GeneralizedHarmonicGaugeSourceFunctions
    IO
    Informer
    Interpolation
    LinearOperators
    MathFunctions
    Time
    Utilities
    ${SPECTRE_LIBRARIES}
  )

function(add_generalized_harmonic_executable
  EXECUTABLE_NAME INITIAL_GUESS BOUNDARY_CONDITIONS EXTRA_LIBS_TO_LINK)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}"
    EvolveGeneralizedHarmonic
    Evolution/Executables/GeneralizedHarmonic
    "EvolutionMetavars<${INITIAL_GUESS}, ${BOUNDARY_CONDITIONS}>"
    "${LIBS_TO_LINK};${EXTRA_LIBS_TO_LINK}"
    )
endfunction(add_generalized_harmonic_executable)

add_generalized_harmonic_executable(
  KerrSchild
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>
  "GeneralRelativitySolutions"
)
