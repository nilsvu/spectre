// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace py = pybind11;

namespace elliptic::dg::py_bindings {

/// An element in a DG domain
template <typename System, bool Linearized, size_t Dim = System::volume_dim>
using DgElement =
    tuples::tagged_tuple_from_typelist<tmpl::remove_duplicates<tmpl::append<
        tmpl::list<
            domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
            domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
            domain::Tags::Coordinates<Dim, Frame::Inertial>,
            domain::Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Inertial>,
            domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
            domain::Tags::Faces<
                Dim, domain::Tags::Coordinates<Dim, Frame::Inertial>>,
            domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>,
            domain::Tags::Faces<
                Dim, domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>>,
            domain::Tags::Faces<
                Dim, domain::Tags::DetSurfaceJacobian<Frame::ElementLogical,
                                                      Frame::Inertial>>,
                                                      domain::Tags::Faces<
                Dim, ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<Dim>,
                                   tmpl::size_t<Dim>, Frame::Inertial>>,
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
            ::Tags::Mortars<domain::Tags::DetSurfaceJacobian<
                                Frame::ElementLogical, Frame::Inertial>,
                            Dim>>,
        typename System::background_fields,
        typename System::fluxes_computer::argument_tags,
        typename elliptic::get_sources_computer<System,
                                                Linearized>::argument_tags,
        domain::make_faces_tags<
            Dim,
            tmpl::remove_duplicates<
                tmpl::append<typename System::background_fields,
                             typename System::fluxes_computer::argument_tags>>,
            typename System::fluxes_computer::volume_tags>>>>;

template <typename System, bool Linearized, size_t Dim = System::volume_dim>
using DgElementArray =
    std::unordered_map<ElementId<Dim>, DgElement<System, Linearized>>;

}  // namespace elliptic::dg::py_bindings
