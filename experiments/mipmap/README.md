SPDX-License-Identifier: Apache-2.0

Running the example:
python main.py

The example requires slangpy:
pip install slangpy

The purpose of this example is to render BRDF using input material and
normal map textures.

This can be expected to give correct results at full res, but when using
lower res texture inputs (mipmap levels) the result will be somewhat
incorrect, represented as per-pixel L2 loss values, which tell us how
the inputs need to change to give the correct values.

This example trains the normal and roughness maps so that the rendered
output BRDF looks as closs as possible to the downsampled original.

The rendered image shows four outputs:
1. The current renderered BRDF with a random light and view direction,
   using the downsampled albedo, and trained normal and roughness.
2. The difference between the L2 loss values of the trained result
   vs using the lower res texture inputs. Green pixels indicate a
   better result and red pixels indicate a worse result.
3. The trained normal map.
4. The trained roughness map.
