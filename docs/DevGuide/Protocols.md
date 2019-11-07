\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Protocols {#protocols}

Protocols are a concept we use in SpECTRE to define metaprogramming interfaces.
A variation of this concept is built into many languages, so this is a quote
from the [Swift documentation](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html):

> A protocol defines a blueprint of methods, properties, and other requirements
> that suit a particular task or piece of functionality. The protocol can then
> be adopted by a class, structure, or enumeration to provide an actual
> implementation of those requirements. Any type that satisfies the requirements
> of a protocol is said to conform to that protocol.

A related feature is proposed for C++20 and goes under the name of
[concepts and constraints](https://en.cppreference.com/w/cpp/language/constraints).
Once this feature becomes available in SpECTRE we can consider transitioning to
it. Until then, our implementation is a combination of structs that define a
protocol through metaprogramming or documentation, and unit tests that enforce
conformance to these protocols.

You should define a protocol when you need a template parameter to conform to an
interface. Here is an example of a protocol that is taken from the
[Swift documentation](https://docs.swift.org/swift-book/LanguageGuide/Protocols.html):

\snippet Utilities/Test_ProtocolHelpers.cpp named_protocol

The protocol defines an interface that any type that adopts it must implement.
For example, the following class conforms to the protocol we just defined:

\snippet Utilities/Test_ProtocolHelpers.cpp named_conformance

Once you have defined a protocol, you can check if a class conforms to it using
the `conforms_to` metafunction:

\snippet Utilities/Test_ProtocolHelpers.cpp conforms_to

So now you can write code that relies on the interface defined by the protocol:

\snippet Utilities/Test_ProtocolHelpers.cpp using_named_protocol

Checking for protocol conformance here makes it clear that we are expecting
a template parameter that exposes the particular interface we have defined in
the protocol. Therefore, the author of the protocol and of the code that uses it
has explicitly defined (and documented!) the interface they expect. And the
developer who consumes the protocol by writing classes that conform to it knows
exactly what needs to be implemented.

We typically define protocols in a file named `Protocols.hpp` and within a
`protocols` namespace, similar to how we write \ref DataBoxTagsGroup "tags" in a
`Tags.hpp` file and within a `Tags` namespace. The file should be placed in the
in the directory associated with the code that depends on classes conforming to
the protocols. For example, the protocol `Named` in the example above would be
placed in directory that also has the `greet` function.

## Protocol users: Testing protocol conformance

Any class that indicates it conforms to the protocol must test that it actually
does using the `test_protocol_conformance` metafunction from
`tests/Unit/ProtocolTestHelpers.hpp`:

\snippet Utilities/Test_ProtocolHelpers.cpp test_protocol_conformance

## Protocol authors: Protocols should not be templates

When you author a new protocol, keep in mind that protocols should typically not
be templates. Instead of adding template parameters to the protocol, make them
part of your protocol instead. The reason for this guideline is that protocols
will always be used as base classes. Therefore, any template parameters of the
protocol must also be template parameters of their conforming classes, which
means the protocol can just check them.

For example, we could be tempted to follow this antipattern:

\snippet Utilities/Test_ProtocolHelpers.cpp named_antipattern

However, instead of making the protocol a template we should add a requirement
to it:

\snippet Utilities/Test_ProtocolHelpers.cpp named_with_type

Classes would need to specify the template parameters for any protocols they
conform to anyway, if the protocols had any. So they might as well expose them:

\snippet Utilities/Test_ProtocolHelpers.cpp person_with_name_type

This pattern also allows us to check for protocol conformance first and then add
further checks about the types if we wanted to:

\snippet Utilities/Test_ProtocolHelpers.cpp example_check_name_type

## Protocol authors: Testing a protocol

In addition to thoroughly documenting the protocol, we require that every
protocol provides a `constexpr bool is_conforming_v` that takes a type as
template parameter and checks if it conforms to the protocol in a
SFINAE-friendly way. This is the example implementation from above:

\snippet Utilities/Test_ProtocolHelpers.cpp named_protocol

It uses the `CREATE_IS_CALLABLE` macro from `Utilities/TypeTraits.hpp` to make
testing member functions easier.

We are currently testing protocol conformance as part of our unit tests, so
that the global `conforms_to` convenience metafunction only needs to check if a
type inherits off the protocol, but doesn't need to check the protocol's
(possibly fairly expensive) implementation of `is_conforming_v`.
This is primarily to keep compile times low, and may be reconsidered when
transitioning to C++ "concepts". Full protocol conformance is tested in the
`test_protocol_conformance` metafunction mentioned above.

To make sure their protocol functions correctly, protocol authors must test
its `is_conforming_v` implementation in a unit test (e.g. in a
`Test_Protocols.hpp`):

\snippet Utilities/Test_ProtocolHelpers.cpp testing_a_protocol

They should make sure to test the implementation with classes that conform to
the protocol, and others that don't. This means the test will always include an
example implementation of a class that conforms to the protocol, and the
protocol author should add it to the documentation of the protocol through a
Doxygen snippet. This gives users a convenient way to see how the author intends
their interface to be implemented.
