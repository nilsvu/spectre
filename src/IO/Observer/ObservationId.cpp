// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ObservationId.hpp"

#include <ostream>
#include <pup.h>

namespace observers {
ObservationKey::ObservationKey(const std::string& tag) noexcept
    : key_(std::hash<std::string>{}(tag)) {}

void ObservationKey::pup(PUP::er& p) noexcept { p | key_; }

bool operator==(const ObservationKey& lhs, const ObservationKey& rhs) noexcept {
  return lhs.key() == rhs.key();
}

bool operator!=(const ObservationKey& lhs, const ObservationKey& rhs) noexcept {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const ObservationKey& t) noexcept {
  return os << '(' << t.key() << ')';
}

ObservationId::ObservationId(const double t, const std::string& tag) noexcept
    : observation_key_(tag),
      combined_hash_([&t](size_t type_hash) {
        size_t combined = type_hash;
        boost::hash_combine(combined, t);
        return combined;
      }(observation_key_.key())),
      value_(t) {}

void ObservationId::pup(PUP::er& p) noexcept {
  p | observation_key_;
  p | combined_hash_;
  p | value_;
}

bool operator==(const ObservationId& lhs, const ObservationId& rhs) noexcept {
  return lhs.hash() == rhs.hash() and lhs.value() == rhs.value();
}

bool operator!=(const ObservationId& lhs, const ObservationId& rhs) noexcept {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const ObservationId& t) noexcept {
  return os << '(' << t.observation_key() << "," << t.hash() << ',' << t.value()
            << ')';
}
}  // namespace observers
