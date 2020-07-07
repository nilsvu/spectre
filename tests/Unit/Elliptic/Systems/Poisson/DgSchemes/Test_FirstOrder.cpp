// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Elliptic/Systems/Poisson/DgSchemes/FirstOrder.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

namespace helpers = TestHelpers::Poisson::dg;

using field_tag = helpers::field_tag;
template <size_t Dim>
using field_gradient_tag = helpers::field_gradient_tag<Dim>;
using operator_applied_to_field_tag = helpers::operator_applied_to_field_tag;
template <size_t Dim>
using operator_applied_to_field_gradient_tag =
    helpers::operator_applied_to_field_gradient_tag<Dim>;

template <size_t Dim>
void test_first_order_operator(
    const ElementId<Dim>& element_id, const DomainCreator<Dim>& domain_creator,
    const std::unordered_map<ElementId<Dim>, helpers::Vars<Dim>>& all_vars,
    const helpers::OperatorVars<Dim>& expected_result,
    const double penalty_parameter) {
  CAPTURE(Dim);
  CAPTURE(element_id);
  CAPTURE(penalty_parameter);
  CAPTURE(all_vars);

  const auto dg_elements =
      TestHelpers::elliptic::dg::create_elements(domain_creator);
  const auto result = helpers::apply_first_order_operator<Dim>(
      element_id, dg_elements, all_vars, penalty_parameter);
  CHECK_VARIABLES_APPROX(result, expected_result);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Poisson.DgSchemes.FirstOrder", "[Unit][Elliptic]") {
  // In these tests we construct random variables for an element and its
  // neighbors, apply the DG operator and then compare the result to hard-coded
  // values. These values were checked against a reference implementation. They
  // are sensitive to many internals of the DG operator implementation such as
  // differentiation operations, Jacobians, lifting operations and numerical
  // fluxes. These tests achieve two goals:
  //
  // 1. They can be used to check against a reference implementation.
  // 2. They assure that changes (e.g. optimizations) anywhere in the operator
  //    implementation don't introduce bugs that change the operator.
  //
  // Note that the data layout in the DataVectors corresponds to the order 'F'
  // in Numpy.
  {
    INFO("1D");
    const domain::creators::Interval domain_creator{
        {{0.}}, {{M_PI}}, {{false}}, {{1}}, {{3}}};
    const double penalty_parameter = 6.75;
    const ElementId<1> element_id{0, {{{1, 0}}}};
    std::unordered_map<ElementId<1>, helpers::Vars<1>> all_vars{};
    auto& element_vars = all_vars.emplace(element_id, 3).first->second;
    get(get<field_tag>(element_vars)) =
        DataVector{0.6964691855978616, 0.28613933495037946, 0.2268514535642031};
    get<0>(get<field_gradient_tag<1>>(element_vars)) =
        DataVector{0.5513147690828912, 0.7194689697855631, 0.42310646012446096};
    const ElementId<1> neighbor_id{0, {{{1, 1}}}};
    auto& neighbor_vars = all_vars.emplace(neighbor_id, 3).first->second;
    get(get<field_tag>(neighbor_vars)) =
        DataVector{0.9807641983846155, 0.6848297385848633, 0.48093190148436094};
    get<0>(get<field_gradient_tag<1>>(neighbor_vars)) =
        DataVector{0.3921175181941505, 0.3431780161508694, 0.7290497073840416};
    // Construct the expected result for the element
    helpers::OperatorVars<1> expected_result{3};
    get(get<operator_applied_to_field_tag>(expected_result)) =
        DataVector{30.449345205689493, 0.08161994446474846, -16.60011524625603};
    get<0>(get<operator_applied_to_field_gradient_tag<1>>(expected_result)) =
        DataVector{-1.3630731065029345, 1.0184369034526106,
                   -1.1647534033114597};
    test_first_order_operator(element_id, domain_creator, all_vars,
                              expected_result, penalty_parameter);
  }
  {
    INFO("2D");
    const domain::creators::Rectangle domain_creator{
        {{0., 0.}}, {{M_PI, M_PI}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    const double penalty_parameter = 6.75;
    const ElementId<2> element_id{0, {{{1, 0}, {1, 0}}}};
    std::unordered_map<ElementId<2>, helpers::Vars<2>> all_vars{};
    auto& element_vars = all_vars.emplace(element_id, 9).first->second;
    get(get<field_tag>(element_vars)) = DataVector{
        0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
        0.5513147690828912, 0.7194689697855631,  0.42310646012446096,
        0.9807641983846155, 0.6848297385848633,  0.48093190148436094};
    get<0>(get<field_gradient_tag<2>>(element_vars)) = DataVector{
        0.3921175181941505, 0.3431780161508694,  0.7290497073840416,
        0.4385722446796244, 0.05967789660956835, 0.3980442553304314,
        0.7379954057320357, 0.18249173045349998, 0.17545175614749253};
    get<1>(get<field_gradient_tag<2>>(element_vars)) =
        DataVector{0.5315513738418384, 0.5318275870968661, 0.6344009585513211,
                   0.8494317940777896, 0.7244553248606352, 0.6110235106775829,
                   0.7224433825702216, 0.3229589138531782, 0.3617886556223141};
    const ElementId<2> east_id{0, {{{1, 1}, {1, 0}}}};
    auto& east_vars = all_vars.emplace(east_id, 9).first->second;
    get(get<field_tag>(east_vars)) =
        DataVector{0.3427638337743084, 0.3041207890271841, 0.4170222110247016,
                   0.6813007657927966, 0.8754568417951749, 0.5104223374780111,
                   0.6693137829622723, 0.5859365525622129, 0.6249035020955999};
    get<0>(get<field_gradient_tag<2>>(east_vars)) =
        DataVector{0.6746890509878248, 0.8423424376202573,  0.08319498833243877,
                   0.7636828414433382, 0.243666374536874,   0.19422296057877086,
                   0.5724569574914731, 0.09571251661238711, 0.8853268262751396};
    get<1>(get<field_gradient_tag<2>>(east_vars)) =
        DataVector{0.6272489720512687, 0.7234163581899548, 0.01612920669501683,
                   0.5944318794450425, 0.5567851923942887, 0.15895964414472274,
                   0.1530705151247731, 0.6955295287709109, 0.31876642638187636};
    const ElementId<2> south_id{0, {{{1, 0}, {1, 1}}}};
    auto& south_vars = all_vars.emplace(south_id, 9).first->second;
    get(get<field_tag>(south_vars)) = DataVector{
        0.22826323087895561, 0.29371404638882936, 0.6309761238544878,
        0.09210493994507518, 0.43370117267952824, 0.4308627633296438,
        0.4936850976503062,  0.425830290295828,   0.3122612229724653};
    get<0>(get<field_gradient_tag<2>>(south_vars)) =
        DataVector{0.4263513069628082, 0.8933891631171348, 0.9441600182038796,
                   0.5018366758843366, 0.6239529517921112, 0.11561839507929572,
                   0.3172854818203209, 0.4148262119536318, 0.8663091578833659};
    get<1>(get<field_gradient_tag<2>>(south_vars)) =
        DataVector{0.2504553653965067, 0.48303426426270435, 0.985559785610705,
                   0.5194851192598093, 0.6128945257629677,  0.12062866599032374,
                   0.8263408005068332, 0.6030601284109274,  0.5450680064664649};
    // Construct the expected result for the element
    helpers::OperatorVars<2> expected_result{9};
    get(get<operator_applied_to_field_tag>(expected_result)) =
        DataVector{62.99910419360309, 15.131123207487887,  9.475351904359094,
                   29.51887907620356, 0.15877084656911436, -5.717099517702478,
                   68.99695898123761, 12.698378957644197,  -5.2038072737404075};
    get<0>(get<operator_applied_to_field_gradient_tag<2>>(expected_result)) =
        DataVector{
            -1.5222703573916752, 0.642145949817917,  0.3596807846140639,
            -2.1771161524596896, 0.1412978410743168, 0.5779904443170825,
            -2.5728600879401284, 0.5006948535281055, 0.016687320123609573};
    get<1>(get<operator_applied_to_field_gradient_tag<2>>(expected_result)) =
        DataVector{
            -1.5781444398110547, -1.410794822540947,  -0.5701143993160409,
            0.6684439677522476,  0.47064113085371095, 0.44927087375959884,
            1.2470182011134718,  1.411957134593697,   0.08972665920689754};
    test_first_order_operator(element_id, domain_creator, all_vars,
                              expected_result, penalty_parameter);
  }
  {
    INFO("3D");
    const domain::creators::Brick domain_creator{{{0., 0., 0.}},
                                                 {{M_PI, M_PI, M_PI}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{2, 2, 2}}};
    const double penalty_parameter = 6.75;
    const ElementId<3> element_id{0, {{{1, 0}, {1, 0}, {1, 0}}}};
    std::unordered_map<ElementId<3>, helpers::Vars<3>> all_vars{};
    auto& element_vars = all_vars.emplace(element_id, 8).first->second;
    get(get<field_tag>(element_vars)) =
        DataVector{0.6964691855978616, 0.28613933495037946, 0.2268514535642031,
                   0.5513147690828912, 0.7194689697855631,  0.42310646012446096,
                   0.9807641983846155, 0.6848297385848633};
    get<0>(get<field_gradient_tag<3>>(element_vars)) =
        DataVector{0.48093190148436094, 0.3921175181941505, 0.3431780161508694,
                   0.7290497073840416,  0.4385722446796244, 0.05967789660956835,
                   0.3980442553304314,  0.7379954057320357};
    get<1>(get<field_gradient_tag<3>>(element_vars)) =
        DataVector{0.18249173045349998, 0.17545175614749253, 0.5315513738418384,
                   0.5318275870968661,  0.6344009585513211,  0.8494317940777896,
                   0.7244553248606352,  0.6110235106775829};
    get<2>(get<field_gradient_tag<3>>(element_vars)) =
        DataVector{0.7224433825702216,  0.3229589138531782,  0.3617886556223141,
                   0.22826323087895561, 0.29371404638882936, 0.6309761238544878,
                   0.09210493994507518, 0.43370117267952824};
    const ElementId<3> east_id{0, {{{1, 1}, {1, 0}, {1, 0}}}};
    auto& east_vars = all_vars.emplace(east_id, 8).first->second;
    get(get<field_tag>(east_vars)) =
        DataVector{0.8069686841371791,  0.3943700539527748, 0.7310730358445571,
                   0.16106901442921484, 0.6006985678335899, 0.8658644583032646,
                   0.9835216092035556,  0.07936579037801572};
    get<0>(get<field_gradient_tag<3>>(east_vars)) = DataVector{
        0.42834727470094924, 0.2045428595464277,  0.4506364905187348,
        0.547763572628854,   0.09332671036982076, 0.29686077548067946,
        0.9275842401521475,  0.5690037314301953};
    get<1>(get<field_gradient_tag<3>>(east_vars)) =
        DataVector{0.45741199752361195, 0.7535259907981146, 0.7418621518420373,
                   0.04857903284426879, 0.708697395442746,  0.8392433478050836,
                   0.16593788420695388, 0.7809979379999573};
    get<2>(get<field_gradient_tag<3>>(east_vars)) =
        DataVector{0.2865366167291019,  0.3064697533295573, 0.665261465349683,
                   0.11139217160771575, 0.6648724488032943, 0.8878567926762226,
                   0.6963112682354063,  0.44032787666540907};
    const ElementId<3> south_id{0, {{{1, 0}, {1, 1}, {1, 0}}}};
    auto& south_vars = all_vars.emplace(south_id, 8).first->second;
    get(get<field_tag>(south_vars)) =
        DataVector{0.8423424376202573,  0.08319498833243877, 0.7636828414433382,
                   0.243666374536874,   0.19422296057877086, 0.5724569574914731,
                   0.09571251661238711, 0.8853268262751396};
    get<0>(get<field_gradient_tag<3>>(south_vars)) =
        DataVector{0.6272489720512687, 0.7234163581899548, 0.01612920669501683,
                   0.5944318794450425, 0.5567851923942887, 0.15895964414472274,
                   0.1530705151247731, 0.6955295287709109};
    get<1>(get<field_gradient_tag<3>>(south_vars)) =
        DataVector{0.31876642638187636, 0.6919702955318197, 0.5543832497177721,
                   0.3889505741231446,  0.9251324896139861, 0.8416699969127163,
                   0.35739756668317624, 0.04359146379904055};
    get<2>(get<field_gradient_tag<3>>(south_vars)) =
        DataVector{0.30476807341109746, 0.398185681917981,   0.7049588304513622,
                   0.9953584820340174,  0.35591486571745956, 0.7625478137854338,
                   0.5931769165622212,  0.6917017987001771};
    const ElementId<3> back_id{0, {{{1, 0}, {1, 0}, {1, 1}}}};
    auto& back_vars = all_vars.emplace(back_id, 8).first->second;
    get(get<field_tag>(back_vars)) =
        DataVector{0.4308627633296438, 0.4936850976503062, 0.425830290295828,
                   0.3122612229724653, 0.4263513069628082, 0.8933891631171348,
                   0.9441600182038796, 0.5018366758843366};
    get<0>(get<field_gradient_tag<3>>(back_vars)) =
        DataVector{0.6239529517921112,  0.11561839507929572, 0.3172854818203209,
                   0.4148262119536318,  0.8663091578833659,  0.2504553653965067,
                   0.48303426426270435, 0.985559785610705};
    get<1>(get<field_gradient_tag<3>>(back_vars)) =
        DataVector{0.5194851192598093, 0.6128945257629677, 0.12062866599032374,
                   0.8263408005068332, 0.6030601284109274, 0.5450680064664649,
                   0.3427638337743084, 0.3041207890271841};
    get<2>(get<field_gradient_tag<3>>(back_vars)) =
        DataVector{0.4170222110247016, 0.6813007657927966, 0.8754568417951749,
                   0.5104223374780111, 0.6693137829622723, 0.5859365525622129,
                   0.6249035020955999, 0.6746890509878248};
    // Construct the expected result for the element
    helpers::OperatorVars<3> expected_result{8};
    get(get<operator_applied_to_field_tag>(expected_result)) =
        DataVector{33.56237429154751,  5.518270647464505,  3.0881452561073126,
                   12.701366149374499, 26.642626648613494, 5.141812735652094,
                   27.996561432519712, 3.7976563925058606};
    get<0>(get<operator_applied_to_field_gradient_tag<3>>(expected_result)) =
        DataVector{-0.14461611119350626, 0.3217713525873183,
                   -0.1522179873748645,  0.408052278449901,
                   -0.28881386542248644, 0.1352894828642397,
                   -0.6623057776799018,  0.7362399834841582};
    get<1>(get<operator_applied_to_field_gradient_tag<3>>(expected_result)) =
        DataVector{-0.40531244467214345, -0.3576880849304651,
                   0.43868557734482205,  0.661025970844067,
                   -0.44800109394098975, 0.14409770345765616,
                   1.0588373196788439,   0.5159440310408107};
    get<2>(get<operator_applied_to_field_gradient_tag<3>>(expected_result)) =
        DataVector{-0.17897084359654736, -0.128560982760031,
                   -0.4070033458288713,  -0.5586908042059262,
                   0.46280434647259,     0.4988483876486043,
                   -0.03456892180097798, 0.5858873867880108};
    test_first_order_operator(element_id, domain_creator, all_vars,
                              expected_result, penalty_parameter);
  }
}
