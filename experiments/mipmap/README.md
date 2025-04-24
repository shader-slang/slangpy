SPDX-License-Identifier: Apache-2.0

Running the example:
python main.py

The example requires slangpy:
pip install slangpy

The purpose of this example is to render pixels using input material and
normal map textures, and calculate per-pixel lighting.

This can be expected to give correct results at full res, but when using
lower res texture inputs (mipmap levels) the result will be somewhat
incorrect, represented as per-pixel L2 loss values, which tell us how
the inputs need to change to give the correct values.

This app has 5 modes, set by the mode global variable in main.py:
1. Evaluate pixel color using full res textures. This simply reads full res
   texture data and evaluates a simple BRDF (in this case phone).
2. Evaluate a lower res pixel color by using mode (1) to evaluate 4 adjacent
   values and take the average. This is similar to using lower res mipmap
   levels but should give a more accurate result.
3. Evaluate a lower res pixel by using mode (1) with lower res texture
   inputs, ie, lower res mipmap levels. This should be less accurate
   than mode 2.
4. Show the difference between mode (1) and (3), ie the per-pixel
   L2 loss values between full res and low res inputs.
5. Train per-pixel loss values to reach the expected values from (4).
   This is shown by rendering the learned loss values as white pixels,
   which should match the values from mode (4).
