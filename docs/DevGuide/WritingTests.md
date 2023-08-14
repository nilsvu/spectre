\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Unit Tests {#writing_unit_tests}

\tableofcontents

Unit tests are placed in the appropriate subdirectory of `tests/Unit`, which
mirrors the directory hierarchy of `src`. Typically there should be one test
executable for each production code library. For example,
we have a `DataStructures` library and a `Test_DataStructures` executable. When
adding a new test there are several scenarios that can occur, which are outlined
below.

- You are adding a new source file to an existing test library:<br>
  If you are adding a new source file in a directory that already has a
  `CMakeLists.txt` simply create the source file, which should be named
  `Test_ProductionCodeFileBeingTest.cpp` and add that to the `LIBRARY_SOURCES`
  in the `CMakeLists.txt` file in the same directory you are adding the `cpp`
  file.<br>
  If you are adding a new source file to a library but want to place it in a
  subdirectory you must first create the subdirectory. To provide a concrete
  example, say you are adding the directory `TensorEagerMath` to
  `tests/Unit/DataStructures`. After creating the directory you must add a call
  to `add_subdirectory(TensorEagerMath)` to
  `tests/Unit/DataStructures/CMakeLists.txt` *before* the call to
  `add_test_library` and *after* the `LIBRARY_SOURCES` are set. Next add the
  file `tests/Unit/DataStructures/TensorEagerMath/CMakeLists.txt`, which should
  add the new source files by calling `set`, e.g.
  ```
  set(LIBRARY_SOURCES
      ${LIBRARY_SOURCES}
      Test_ProductionCodeFileBeingTest.cpp
      PARENT_SCOPE)
  ```
  The `PARENT_SCOPE` flag tells CMake to make the changes visible in the
  CMakeLists.txt file that called `add_subdirectory`. You can now add the
  `Test_ProductionCodeFileBeingTested.cpp` source file.
- You are adding a new directory:<br>
  If the directory is a new lowest level directory you must add a
  `add_subdirectory` call to `tests/Unit/CMakeLists.txt`. If it is a new
  subdirectory you must add a `add_subdirectory` call to the
  `CMakeLists.txt` file in the directory where you are adding the
  subdirectory. Next you should read the part on adding a new test library.
- You are adding a new test library:<br>
  After creating the subdirectory for the new test library you must add a
  `CMakeLists.txt` file. See `tests/Unit/DataStructures/CMakeLists.txt` for
  an example of one. The `LIBRARY` and `LIBRARY_SOURCES` variables set the name
  of the test library and the source files to be compiled into it. The library
  name should be of the format `Test_ProductionLibraryName`, for example
  `Test_DataStructures`. The library sources should be only the source files in
  the current directory. The `add_subdirectory` command can be used to add
  source files in subdirectories to the same library as is done in
  `tests/Unit/CMakeLists.txt`. The `CMakeLists.txt` in
  `tests/Unit/DataStructures/TensorEagerMath` is an example of how to add source
  files to a library from a subdirectory of the library. Note that the setting
  of `LIBRARY_SOURCES` here first includes the current `LIBRARY_SOURCES` and at
  the end specifies `PARENT_SCOPE`. The `PARENT_SCOPE` flag tells CMake to
  modify the variable in a scope that is visible to the parent directory,
  i.e. the `CMakeLists.txt` that called `add_subdirectory`.<br>
  Finally, in the `CMakeLists.txt` of your new library you must call
  `add_test_library`. Again, see `tests/Unit/DataStructures/CMakeLists.txt` for
  an example. The `add_test_library` function adds a test executable with the
  name of the first argument and the source files of the third argument. The
  second and fourth arguments are unused for historical reasons and will be
  removed in the future. Remember to use `target_link_libraries` to link any
  libraries your test executable uses (see \ref spectre_build_system).

All tests must start with
```cpp
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"
```
The file `tests/Unit/Framework/TestingFramework.hpp` must always be the first
include in the test file and must be separated from the STL includes by a blank
line. All classes and free functions should be in an anonymous/unnamed
namespace, e.g.
```cpp
namespace {
class MyFreeClass {
  /* ... */
};

void my_free_function() {
  /* ... */
}
}  // namespace
```
This is necessary to avoid symbol redefinition errors during linking.

Test cases are added by using the `SPECTRE_TEST_CASE` macro. The first argument
to the macro is the test name, e.g. `"Unit.DataStructures.Tensor"`, and the
second argument is a list of tags. The tags list is a string where each element
is in square brackets. For example, `"[Unit][DataStructures]"`. The tags should
only be the type of test, in this case `Unit`, and the library being tested, in
this case `DataStructures`. The `SPECTRE_TEST_CASE` macro should be treated as a
function, which means that it should be followed by `{ /* test code */ }`. For
example,
\snippet Test_Tensor.cpp example_spectre_test_case
From within a `SPECTRE_TEST_CASE` you are able to do all the things you would
normally do in a C++ function, including calling other functions, setting
variables, using lambdas, etc.

The `CHECK` macro in the above example is provided by
[Catch2](https://github.com/catchorg/Catch2) and is used to check conditions. We
also provide the `CHECK_ITERABLE_APPROX` macro which checks if two `double`s or
two iterable containers of `double`s are approximately
equal. `CHECK_ITERABLE_APPROX` is especially useful for comparing `Tensor`s,
`DataVector`s, and `Tensor<DataVector>`s since it will iterate over nested
containers as well.

\warning Catch's `CHECK` statement only prints numbers out to approximately 10
digits at most, so you should generally prefer `CHECK_ITERABLE_APPROX` for
checking double precision numbers, unless you want to check that two numbers are
bitwise identical.

All unit tests must finish within a few seconds, the hard limit is 5, but having
unit tests that long is strongly discouraged. They should typically complete in
less than half a second. Tests that are longer are often no longer testing a
small enough unit of code and should either be split into several unit tests or
moved to an integration test.

#### Discovering New and Renamed Tests

When you add a new test to a source file or rename an existing test the change
needs to be discovered by the testing infrastructure. This is done by building
the target `rebuild_cache`, e.g. by running `make rebuild_cache`.

#### Testing Pointwise Functions

Pointwise functions should generally be tested in two different ways. The first
is by taking input from an analytic solution and checking that the computed
result is correct. The second is to use the random number generation comparison
with Python infrastructure. In this approach the C++ function being tested is
re-implemented in Python and the results are compared. Please follow these
guidelines:

- The Python implementation should be in a file with the same name as the source
  file that is being re-implemented and placed in the same directory as its
  corresponding `Test_*.cpp` source file.
- The functions should have the same names as the C++ functions they
  re-implement.
- If a function does sums over tensor indices then
  [`numpy.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html)
  should be used in Python to provide an alternative implementation of the loop
  structure.
- You can import Python functions from other re-implementations in the
  `tests/Unit/` directory to reduce code duplication. Note that the path you
  pass to `pypp::SetupLocalPythonEnvironment` determines the directory from
  which you can import Python modules. Either import modules directly from the
  `tests/Unit/` directory (e.g. `import
  PointwiseFunction.GeneralRelativity.Christoffel as christoffel`) or use
  relative imports like `from . import Christoffel as christoffel`. Don't assume
  the Python environment is set up in a subdirectory of `tests/Unit/`.

It is possible to test C++ functions that return by value and ones that return
by `gsl::not_null`. In the latter case, since it is possible to return multiple
values, one Python function taking all non-`gsl::not_null` arguments must be
supplied for each `gsl::not_null` argument to the C++. To perform the test the
`pypp::check_with_random_values()` function must be called. For example, the
following checks various C++ functions by calling into `pypp`:

\snippet Test_PyppRandomValues.cpp cxx_two_not_null

The corresponding Python functions are:

\snippet PyppPyTests.py python_two_not_null

#### Writing and Fixing Random-Value Based Tests

Many tests in SpECTRE make use of randomly generated numbers in order to
increase the parameter space covered by the tests. The random number generator
is set up using:
```cpp
MAKE_GENERATOR(gen);
```
The generator `gen` can then be passed to distribution classes such as
`std::uniform_real_distribution` or `UniformCustomDistribution`.

Each time the test is run, a different random seed will be used.  When writing a
test that uses random values, it is good practice to run the test at least
\f$10^4\f$ times in order to set any tolerances on checks used in the test.
This can be done by using the following command in the build directory
(SPECTRE_BUILD_DIR):
```
ctest --repeat-until-fail 10000 -R TEST_NAME
```
where `TEST_NAME` is the test name passed to `SPECTRE_TEST_CASE`
(e.g. `Unit.Evolution.Systems.CurvedScalarWave.Characteristics`).

If a test case fails when using a random number generated by `MAKE_GENERATOR`,
as part of the output from the failed test will be the text
```
Seed is:  SEED from FILE_NAME:LINE_NUMBER
```
Note that the output of tests can be found in
`SPECTRE_BUILD_DIR/Testing/Temporary/LastTest.log`

The failing test case can then be reproduced by changing `MAKE_GENERATOR` call
at the provided line in the given file to
```cpp
MAKE_GENERATOR(gen, SEED);
```
If the `MAKE_GENERATOR` is within `CheckWithRandomValues.hpp`, the failing test
case most likely has occurred within a call to
`pypp::check_with_random_values()`.  In such a case, additional information
should have been printed to help you determine which call to
`pypp::check_with_random_values()` has failed.  The critical information is
the line
```
function:  FUNCTION_NAME
```
where `FUNCTION_NAME` should correspond to the third argument of a call to
`pypp::check_with_random_values()`.  The seed that caused the test to fail can
then be passed as an additional argument to `pypp::check_with_random_values()`,
where you may also need to pass in the default value of the comparison
tolerance.

Typically, you will need to adjust a tolerance used in a `CHECK` somewhere in
the test in order to get the test to succeed reliably.  The function
`pypp::check_with_random_values()` takes an argument that specifies the lower
and upper bounds of random quantities.  Typically these should be chosen to be
of order unity in order to decrease the chance of occasionally generating large
numbers through multiplications which can cause an error above a reasonable
tolerance.

#### Testing Failure Cases {#testing_failure_cases}

Adding the "attribute" `// [[OutputRegex, Regular expression to
match]]` before the `SPECTRE_TEST_CASE` macro will force ctest to only
pass the particular test if the regular expression is found in the
output of the test. This can be used to test error handling. When
testing `ASSERT`s you must mark the `SPECTRE_TEST_CASE` as
`[[noreturn]]`, add the macro `ASSERTION_TEST();` to the beginning of
the test, and also have the test call `ERROR("Failed to trigger ASSERT
in an assertion test");` at the end of the test body.  The test body
should be enclosed between `#%ifdef SPECTRE_DEBUG` and an `#%endif`

If the `#%ifdef SPECTRE_DEBUG` block is omitted then compilers will
correctly flag the code as being unreachable which results in
warnings.

You can also test `ERROR`s inside your code. These tests need to have
the `OutputRegex`, and also call `ERROR_TEST();` at the
beginning. They do not need the `#%ifdef SPECTRE_DEBUG` block, they
can just call have the code that triggers an `ERROR`.

We are currently transforming these failure cases to use the
`CHECK_THROWS_WITH` macro. This macro takes two arguments: the first
is either an expression or a lambda that is expected to trigger an
exception (which now are thrown by `ASSERT` and `ERROR` (Note: You may
need to add `()` wrapping the lambda in order for it to compile.); the
second is a Catch Matcher (see
[Catch2](https://github.com/catchorg/Catch2) for complete
documentation), usually a `Catch::Matchers::ContainsSubstring()` macro
that matches a substring of the error message of the thrown exception.

Note that a `OutputRegex` can also be specified in a test that is
supposed to succeed with output that matches the regular expression.
In this case, the first line of the test should call the macro
`OUTPUT_TEST();`.

### Testing Actions

The action testing framework is documented as part of the `ActionTesting`
namespace.

## Input file tests

We have a suite of input file tests in addition to unit tests. Every input file
in the `tests/InputFiles/` directory is added to the test suite automatically.
The input file must specify the `Executable` it should run with in the input
file metadata (above the `---` marker in the input file). Properties of the test
are controlled by the `Testing` section in the input file metadata. The
following properties are available:

- `Check`: Semicolon-separated list of checks, e.g. `parse;execute`. The
  following checks are available:
    - `parse`: Just check that the input file passes option parsing.
    - `execute`: Run the executable. If the input file metadata has an
      `ExpectedOutput` field, check that these files have been written. See
      `spectre.tools.CleanOutput` for details.
    - `execute_check_output`: In additional to `execute`, check the contents of
      some output files. The checks are defined by the `OutputFileChecks` in the
      input file metadata. See `spectre.tools.CheckOutputFiles` for details.
- `CommandLineArgs` (optional): Additional command-line arguments passed to the
  executable.
- `ExpectedExitCode` (optional): The expected exit code of the executable.
  Default: `0`. See `Parallel::ExitCode` for possible exit codes.
- `Timeout` (optional): Timeout for the test. Default: 2 seconds.
