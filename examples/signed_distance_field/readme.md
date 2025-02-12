# Signed Distance Field Generator

This example implements the Eikonal Sweep method of generating distance fields
as described in Chris Cumming's blog and code here:

- https://shaderfun.com/
- https://github.com/chriscummings100/signeddistancefields

## Overview

The example takes a greyscale image as input and:

1. Computes a signed distance field (negative inside shapes, positive outside)
2. Generates isolines visualization with:
   - Green lines marking shape boundaries
   - Red isolines showing distance outside shapes
   - Blue isolines showing distance inside shapes

## Input Example

![Input binary image](input.png)

## Output Example

![Output visualization](output.png)

_Visualization showing the distance field with isolines. Green lines mark shape boundaries, red lines show distances outside shapes, and blue lines show distances inside._

## Usage

Call the main script either with an example image as input or with nothing to
use a built in example. Input images are passed with `--input` or `-i`.

The output is visualized with `tev`, so make sure that's running.
