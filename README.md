# PyramidalWaveletDecomposition

This is a Pixinsight script for testing out a Multiscale Wavelet Decomposition library that I created. I intend to use the library for multiscale filtering of astronomical images, but I wanted to share the library for others to use.

Essentially, the script takes an image and decomposes it to the number of levels requested.

Approximately 10 different wavelet functions have been implemented, and the test script permits the user to select from the implemented functions.

The test script decomposes the image to the requested number of levels and then recomposes the image to the original. There seems to be very little error associated with the decomposition/recomposition steps.

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
