# SPDX-License-Identifier: Apache-2.0

import sgl
import slangpy as spy
import pathlib
import numpy as np
from PIL import Image
import cv2
import argparse
import sys

def create_image(width=256, height=256, scale=1):
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
        scale (int): Resolution multiplier for antialiasing. If > 1, the image is created
                    at a higher resolution and then downsampled for antialiasing.

    Returns:
        numpy.ndarray: A floating-point RGBA image array with values normalized to [0,1]
    """
    antialiased = scale > 1
    image = np.zeros((height * scale, width * scale, 4), dtype=np.uint8)
    image[:,:,3] = 255

    cv2.circle(image, 
               center=(int(0.31 * width * scale), int(0.39 * height * scale)), 
               radius=int(0.16 * min(width, height) * scale), 
               color=(255, 255, 255, 255), 
               thickness=-1, 
               lineType=cv2.LINE_AA if antialiased else cv2.LINE_8)

    cv2.circle(image, 
               center=(int(0.63 * width * scale), int(0.55 * height * scale)), 
               radius=int(0.23 * min(width, height) * scale), 
               color=(255, 255, 255, 255), 
               thickness=-1, 
               lineType=cv2.LINE_AA if antialiased else cv2.LINE_8)

    rect_center = (int(0.59 * width * scale), int(0.43 * height * scale))
    rect_size = (int(0.39 * width * scale), int(0.55 * height * scale))
    angle = 15 # degrees
    rect = cv2.boxPoints((rect_center, rect_size, angle))
    rect = np.int32(rect)

    cv2.polylines(image, 
                  [rect], 
                  isClosed=True, 
                  color=(255, 255, 255, 255), 
                  thickness=int(0.039 * min(width, height) * scale), 
                  lineType=cv2.LINE_AA if antialiased else cv2.LINE_8)

    if scale > 1:
        image = cv2.resize(image, (width, height), 
                          interpolation=cv2.INTER_AREA)

    return image.astype(np.float32) / 255.0

def load_and_process_image(image_path):
    """
    Loads an image from path, converts it to binary and returns it in the required format.
    Preserves the original image dimensions.

    Args:
        image_path (str): Path to the input image

    Returns:
        tuple: (image array, width, height) where image array is a floating-point RGBA 
               image array with values normalized to [0,1]
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Get original dimensions
    height, width = img.shape[:2]

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Convert to RGBA float32
    result = np.zeros((height, width, 4), dtype=np.float32)
    result[:,:,0] = img.astype(np.float32) / 255.0
    result[:,:,1] = result[:,:,0]
    result[:,:,2] = result[:,:,0]
    result[:,:,3] = 1.0

    return result, width, height

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
    input_tex = device.create_texture(
        width=width, height=height,
        format=sgl.Format.rgba32_float,
        usage=sgl.ResourceUsage.shader_resource,
        data=image
    )
    sgl.tev.show(input_tex, name=f'input_{name_suffix}')

    dist_tex = device.create_texture(
        width=width, height=height,
        format=sgl.Format.rg32_float,
        usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access
    )

    # Initialize
    module.init_eikonal(spy.grid((width, height)), input_tex, dist_tex)
    sgl.tev.show(dist_tex, name=f'initial_distances_{name_suffix}')

    for i in range(128):
        module.solve_eikonal(spy.grid((width, height)), dist_tex)

    distances = dist_tex.to_numpy()
    sgl.tev.show(dist_tex, name=f'final_distances_{name_suffix}')

    result = module.generate_isolines.map((0, 1))(distances, _result='numpy')

    output_tex = device.create_texture(
        width=width, height=height,
        format=sgl.Format.rgba32_float,
        usage=sgl.ResourceUsage.shader_resource,
        data=result
    )
    sgl.tev.show(output_tex, name=f'isolines_{name_suffix}')

    return distances

def main():
    """
    Main function that orchestrates the complete image processing pipeline. It:
    1. Sets up the GPU device and loads the Slang shader module
    2. Either processes a provided input image or generates and processes test images
    3. Generates and visualizes distance fields and isolines
    """
    parser = argparse.ArgumentParser(description='Generate distance fields from binary images')
    parser.add_argument('--input', '-i', type=str, help='Path to input image (optional)')
    args = parser.parse_args()

    device = spy.create_device(include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ])

    module = spy.Module.load_from_file(device, "example.slang")

    if args.input:
        try:
            input_image, width, height = load_and_process_image(args.input)
            distances = process_image(device, module, input_image, width, height, "input")
        except Exception as e:
            print(f"Error processing input image: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        width, height = 256, 256  # Default size for test images

        aliased_image = create_image(width, height, scale=1)
        aliased_distances = process_image(device, module, aliased_image, width, height, "aliased")

        antialiased_image = create_image(width, height, scale=4)
        antialiased_distances = process_image(device, module, antialiased_image, width, height, "antialiased")

if __name__ == "__main__":
    main()

