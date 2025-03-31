# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl
import slangpy as spy
import pathlib
import numpy as np
import argparse
import sys
import shape_utils


def create_image(width=256, height=256, antialiased=False):
    """
    Creates a test image containing geometric shapes (circles and a rotated rectangle)
    with optional antialiasing based on the scale parameter.

    More or less this
       ___
      /  /
     O  /
    /__o

    Args:
        width (int): Width of the output image
        height (int): Height of the output image
        antialiased (bool): Draw with analytical coverage

    Returns:
        numpy.ndarray: A floating-point RGBA image array with values normalized to [0,1]
    """
    # Create a bitmap with RGBA format
    bitmap = sgl.Bitmap(pixel_format=sgl.Bitmap.PixelFormat.rgba,
                        component_type=sgl.Struct.Type.uint8,
                        width=width,
                        height=height)

    # Get numpy array view of bitmap data
    image = np.array(bitmap, copy=False)
    image.fill(0)
    image[:, :, 3] = 255  # Set alpha to fully opaque

    white = [255, 255, 255, 255]
    thickness = 0.039 * min(width, height)

    # Draw shapes
    shape_utils.draw_circle(image,
                            center_x=0.31 * width,
                            center_y=0.39 * height,
                            radius=0.16 * min(width, height),
                            color=white,
                            antialiased=antialiased)

    shape_utils.draw_circle(image,
                            center_x=0.63 * width,
                            center_y=0.55 * height,
                            radius=0.23 * min(width, height),
                            color=white,
                            antialiased=antialiased)

    shape_utils.draw_rotated_rect(image,
                                  center_x=0.59 * width,
                                  center_y=0.43 * height,
                                  width=0.39 * width,
                                  height=0.55 * height,
                                  angle=15,
                                  thickness=thickness,
                                  color=white,
                                  antialiased=antialiased)

    return image.astype(np.float32) / 255.0


def load_image(image_path):
    """
    Loads an image from path, converts it to grayscale
    Preserves the original image dimensions.

    Args:
        image_path (str): Path to the input image

    Returns:
        tuple: (image array, width, height) where image array is a
        floating-point RGBA image array
    """
    bitmap = sgl.Bitmap(image_path)

    # Convert to grayscale
    if bitmap.pixel_format != sgl.Bitmap.PixelFormat.rgba:
        bitmap = bitmap.convert(pixel_format=sgl.Bitmap.PixelFormat.rgba)
    image = np.array(bitmap, copy=False)
    gray = np.mean(image[:, :, :3], axis=2)

    # Create final RGBA result
    result = np.zeros((bitmap.height, bitmap.width, 4), dtype=np.float32)
    result[:, :, 0] = gray
    result[:, :, 1] = gray
    result[:, :, 2] = gray
    result[:, :, 3] = 1.0

    return result, bitmap.width, bitmap.height


def process_image(device, module, image, width, height, name_suffix):
    """
    Processes an input image through an Eikonal equation solver to generate
    distance fields. These are then visualised with isolines. This function
    handles the pipeline from input image to final visualization.

    Images are sent to `tev` as the pipeline progresses

    Args:
        device: The GPU device context for computation
        module: The loaded Slang shader module containing the processing kernels
        image (numpy.ndarray): Input RGBA image as floating-point values in [0,1]
        width (int): Width of the input image
        height (int): Height of the input image
        name_suffix (str): Suffix for naming the visualization outputs

    Returns:
        numpy.ndarray: The computed distance field
    """
    input_tex = device.create_texture(width=width,
                                      height=height,
                                      format=sgl.Format.rgba32_float,
                                      usage=sgl.TextureUsage.shader_resource,
                                      data=image)
    sgl.tev.show(input_tex, name=f'input_{name_suffix}')

    dist_tex = device.create_texture(width=width,
                                     height=height,
                                     format=sgl.Format.rg32_float,
                                     usage=sgl.TextureUsage.shader_resource
                                     | sgl.TextureUsage.unordered_access)

    # Initialize
    module.init_eikonal(spy.grid((width, height)), input_tex, dist_tex)
    sgl.tev.show(dist_tex, name=f'initial_distances_{name_suffix}')

    for i in range(128):
        module.solve_eikonal(spy.grid((width, height)), dist_tex)

    distances = dist_tex.to_numpy()
    sgl.tev.show(dist_tex, name=f'final_distances_{name_suffix}')

    result = module.generate_isolines(distances, _result='numpy')

    output_tex = device.create_texture(width=width,
                                       height=height,
                                       format=sgl.Format.rgba32_float,
                                       usage=sgl.TextureUsage.shader_resource,
                                       data=result)
    sgl.tev.show(output_tex, name=f'isolines_{name_suffix}')

    return distances


def main():
    """
    Main function that orchestrates the complete image processing pipeline. It:
    1. Sets up the GPU device and loads the Slang shader module
    2. Either processes a provided input image or generates and processes test images
    3. Generates and visualizes distance fields and isolines
    """
    parser = argparse.ArgumentParser(
        description='Generate distance fields from binary images')
    parser.add_argument('--input',
                        '-i',
                        type=str,
                        help='Path to input image (optional)')
    args = parser.parse_args()

    device = spy.create_device(include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ])

    module = spy.Module.load_from_file(device, "example.slang")

    if args.input:
        try:
            input_image, width, height = load_image(args.input)
            distances = process_image(device, module, input_image, width,
                                      height, "input")
        except Exception as e:
            print(f"Error processing input image: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        width, height = 256, 256  # Default size for test images

        aliased_image = create_image(width, height, antialiased=False)
        aliased_distances = process_image(device, module, aliased_image, width,
                                          height, "aliased")

        antialiased_image = create_image(width, height, antialiased=True)
        antialiased_distances = process_image(device, module,
                                              antialiased_image, width, height,
                                              "antialiased")


if __name__ == "__main__":
    main()
