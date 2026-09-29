// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

int branchHelper(int x)
{
    if (x < 0)
        return -x;
    return x;
}

int wrapperA(int x) { return branchHelper(x); }

int wrapperB(int x)
{
    if (x == 123)
        return 0;
    return branchHelper(x);
}

int overloaded(int x)
{
    if (x < 0)
        return -x;
    return x;
}

float overloaded(float x)
{
    if (x < 0.0)
        return -x;
    return x;
}

int genericBranch<let Bias : int>(int x)
{
    if (x < Bias)
        return Bias - x;
    return x - Bias;
}

int genericZero(int x) { return genericBranch<0>(x); }
int genericTwo(int x) { return genericBranch<2>(x); }
int genericBoth(int x) { return genericBranch<0>(x) + genericBranch<2>(x); }

int sameLine(int x) { if (x < 0) return -1; if (x > 2) return 1; return 0; }

#define ABS_BODY(value) if (value < 0) return -value; else return value;
int macroA(int x) { ABS_BODY(x) }
int macroB(int x) { ABS_BODY(x) }
int macroBoth(int x) { return macroA(x) + macroB(x); }

#define ACCUMULATE(value) if (value < 0) sum += 1; else sum += 2;
int repeatedMacro(int x)
{
    int sum = 0;
    ACCUMULATE(x)
    ACCUMULATE(x + 10)
    return sum;
}

int switchBranch(int x)
{
    switch (x)
    {
    case -1: return 10;
    case 0: return 20;
    case 1: return 30;
    default: return 40;
    }
}
