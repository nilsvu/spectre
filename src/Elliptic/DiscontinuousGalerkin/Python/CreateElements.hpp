// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Elliptic/DiscontinuousGalerkin/Initialization.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/CreateElements.hpp"
#include "Elliptic/DiscontinuousGalerkin/Python/DgElementArray.hpp"

namespace py = pybind11;

namespace elliptic::dg::py_bindings {

template <typename Initializer, typename Box, typename... ReturnTags,
          typename... ArgsTags>
void apply_initializer(const gsl::not_null<Box*> dg_element,
                       const Initializer& initializer,
                       tmpl::list<ReturnTags...> /*meta*/,
                       tmpl::list<ArgsTags...> /*meta*/) {
  initializer(make_not_null(&get<ReturnTags>(*dg_element))...,
              get<ArgsTags>(*dg_element)...);
}

template <typename System, bool Linearized, typename Initializers,
          typename... GlobalCacheTags, size_t Dim = System::volume_dim>
DgElementArray<System, Linearized> create_elements(
    const DomainCreator<Dim>& domain_creator,
    const tuples::TaggedTuple<GlobalCacheTags...>& /* global_cache */ =
        tuples::TaggedTuple<>{}) {
  const auto domain = domain_creator.create_domain();
  const auto initial_refinement = domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();
  const auto element_ids = ::initial_element_ids(initial_refinement);
  DgElementArray<System, Linearized> element_array{};
  for (const auto& element_id : element_ids) {
    DgElement<System, Linearized> dg_element{};
    ElementMap<Dim, Frame::Inertial> element_map{};
    elliptic::dg::InitializeGeometry<Dim>{}(
        make_not_null(&get<domain::Tags::Mesh<Dim>>(dg_element)),
        make_not_null(&get<domain::Tags::Element<Dim>>(dg_element)),
        make_not_null(&element_map),
        make_not_null(
            &get<domain::Tags::Coordinates<Dim, Frame::ElementLogical>>(
                dg_element)),
        make_not_null(
            &get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(dg_element)),
        make_not_null(
            &get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>(dg_element)),
        make_not_null(
            &get<domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                              Frame::Inertial>>(dg_element)),
        initial_extents, initial_refinement, domain, element_id);
    DirectionMap<Dim, Direction<Dim>> unused_face_directions{};
    elliptic::dg::InitializeFacesAndMortars<Dim,
                                            typename System::inv_metric_tag>{}(
        make_not_null(&unused_face_directions),
        make_not_null(
            &get<domain::Tags::Faces<
                Dim, domain::Tags::Coordinates<Dim, Frame::Inertial>>>(
                dg_element)),
        make_not_null(
            &get<domain::Tags::Faces<Dim, domain::Tags::FaceNormal<Dim>>>(
                dg_element)),
        make_not_null(
            &get<domain::Tags::Faces<
                Dim, domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>>>(
                dg_element)),
        make_not_null(
            &get<domain::Tags::Faces<
                Dim, domain::Tags::DetSurfaceJacobian<
                         Frame::ElementLogical, Frame::Inertial>>>(dg_element)),
        make_not_null(
            &get<domain::Tags::Faces<
                Dim, ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<Dim>,
                                   tmpl::size_t<Dim>, Frame::Inertial>>>(
                dg_element)),
        make_not_null(&get<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>(
            dg_element)),
        make_not_null(&get<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>(
            dg_element)),
        make_not_null(
            &get<::Tags::Mortars<domain::Tags::DetSurfaceJacobian<
                                     Frame::ElementLogical, Frame::Inertial>,
                                 Dim>>(dg_element)),
        get<domain::Tags::Mesh<Dim>>(dg_element),
        get<domain::Tags::Element<Dim>>(dg_element), element_map,
        get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                          Frame::Inertial>>(dg_element),
        initial_extents);
    tmpl::for_each<Initializers>([&dg_element](const auto initializer) {
      using Initializer = tmpl::type_from<std::decay_t<decltype(initializer)>>;
      apply_initializer(make_not_null(&dg_element), Initializer{},
                        typename Initializer::return_tags{},
                        typename Initializer::argument_tags{});
    });
    element_array.emplace(element_id, std::move(dg_element));
  }
  return element_array;
}

}  // namespace elliptic::dg::py_bindings
