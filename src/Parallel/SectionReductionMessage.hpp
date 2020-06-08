// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ckmulticast.h"

template <typename SectionProxy, typename TargetProxy, typename ReductionData,
          typename SectionIdTag>
struct SectionReductionMessage
    : public CkMcastBaseMsg,
      public CMessage_SectionReductionMessage<SectionProxy, TargetProxy,
                                              ReductionData, SectionIdTag> {
  SectionProxy section_proxy;
  typename SectionIdTag::type section_id;
  TargetProxy target_proxy;
  ReductionData data;
  SectionReductionMessage(SectionProxy local_section_proxy,
                          typename SectionIdTag::type local_section_id,
                          TargetProxy local_target_proxy,
                          ReductionData local_data)
      : section_proxy(local_section_proxy),
        section_id(local_section_id),
        target_proxy(local_target_proxy),
        data(local_data) {}
  void pup(PUP::er& p) {
    CMessage_SectionReductionMessage<SectionProxy, TargetProxy, ReductionData,
                                     SectionIdTag>::pup(p);
    p | section_proxy;
    p | section_id;
    p | target_proxy;
    p | data;
  }
};
