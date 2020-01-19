// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/NumericInitialData.hpp"

namespace {

struct SomeNumericInitialData : evolution::MarkAsNumericInitialData {};
struct NoNumericInitialData {};

static_assert(evolution::is_numeric_initial_data_v<SomeNumericInitialData>,
              "Failed testing evolution::is_numeric_initial_data_v");
static_assert(not evolution::is_numeric_initial_data_v<NoNumericInitialData>,
              "Failed testing evolution::is_numeric_initial_data_v");

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using variables_tag = ::Tags::Variables<tmpl::list<FieldTag>>;
};

static_assert(cpp17::is_same_v<
                  typename evolution::NumericInitialData<System>::import_fields,
                  tmpl::list<FieldTag>>,
              "Failed testing evolution::NumericInitialData");

}  // namespace
