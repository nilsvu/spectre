// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/NumericData.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

using all_background_spacetimes =
    tmpl::list<WrappedGr<gr::Solutions::KerrSchild>,
               WrappedGr<RelativisticEuler::Solutions::TovStar>, NumericData>;

}  // namespace ray_tracing
