# PyramidalWaveletDecomposition

This is a Pixinsight script for testing out a Multiscale Wavelet Decomposition library that I created from scratch using Pixinsight Image, View and Window structures, and some functions such as Crop, but otherwise from scratch. I intend to use the library to test multiscale filtering of astronomical images, but I wanted to share the library for others to use.

Essentially, the script takes an image and decomposes it to the number of levels requested in the dropdown.

Around 14 different wavelet functions have been implemented at the present time, and the test script permits the user to select the implemented functions from the dropdown.

The test script decomposes the image to the requested number of levels and then recomposes the image to the original. There seems to be very little error associated with the decomposition/recomposition steps. The recomposed image is displayed and should be the same as the initial image.

# Usage

These algorithms likely only work on mono images.

Select Linear data if you wish the algorithm to treat the data as linear, otherwise significant blending artifacts may be seen.

## Uses

The script is used to test the wavelet library, something like a unit test.

## Script

This script can be found here after installation: ChickadeeScripts > DenoisingSuite.

## Manage repository location

In order to automatically install and subsequently refresh script updates in Pixinsight, add the following URL to Resources > Updates > Manage repositories

https://raw.githubusercontent.com/chickadeebird/PyramidalWaveletDecomposition/main/

After this has been added to the repositories, Resources > Updates > Check for updates should place the new PyramidalWaveletDecomposition script in Scripts > ChickadeeScripts

# Immediate Future Work

1. Need to implement the functionality for RGB images.
2. Need to test further for different image sizes.
3. Need to implement a restriction on the number of levels permitted based on the image size and the wavelet filter size.
