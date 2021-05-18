// Distributed under the MIT License.
// See LICENSE.txt for details.

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// Charm checks if this symbols exists in its build system and then references
// it, but it is missing in Python binding libs because they are not compiled as
// executables (at least that's what I think is happening). This upstream issue
// is somewhat related: https://github.com/UIUC-PPL/charm/issues/1893
// char __executable_start = '0';
