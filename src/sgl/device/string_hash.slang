// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helper class to wrap Slang's String class, allowing it to be passed and returned,
// as the `String` itself is supposed to be only really a literal passed directly into
// a function call.
struct StringHash
{
    uint hash;

    __init() { hash = getStringHash(""); }
    __implicit_conversion(1) __init(String str) { hash = getStringHash(str); }
}

bool operator ==(StringHash a, StringHash b)
{
    return a.hash == b.hash;
}
