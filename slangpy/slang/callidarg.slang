// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing slangpy;

public struct CallIdArg {

    public void load<let N : int>(ContextND<N> context, out vector<uint,N> value) {
        var t = context.call_id;
        for (int i = 0; i < N; i++) {
            value[i] = t[i];
        }
    }

}
