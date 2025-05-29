#include "WaveletsDefinitions.js"

// const PADDING_FILL_TYPE = {
#define PADDING_SYMMETRIC 0   // ... x2 x1 | x1 x2 ... xn | xn xn-1 ...
#define PADDING_ZERO 1        // ... 0  0  | x1 x2 ... xn | 0  0 ...
#define PADDING_REFLECT 2     // ... x3 x2 | x1 x2 ... xn | xn-1 xn-2 ...
#define PADDING_CONSTANT 3    // ... x1 x1 | x1 x2 ... xn | xn xn ...

#define HORIZONTAL_AXIS 0
#define VERTICAL_AXIS 1

#define SCALE_FACTOR 10000

// export { PADDING_FILL_TYPE };

function padImage(view, padAmountHorizontal, padAmountVertical) {
    Console.warningln('Prepadded image width: ' + view.image.width + ' and height: ' + view.image.height);

    let paddedImageWindow = new ImageWindow(view.image.width, view.image.height,
        view.image.numberOfChannels, // 1 channel for grayscale
        view.image.bitsPerSample,
        view.image.isReal,
        view.image.isColor
    );

    let paddedImageView = paddedImageWindow.mainView;
    paddedImageView.beginProcess(UndoFlag_NoSwapFile);
    let paddedImage = paddedImageView.image;
    paddedImage.apply(view.image);

    paddedImageWindow.show();

    var PCrop = new Crop;
    PCrop.leftMargin = padAmountHorizontal;
    PCrop.topMargin = padAmountVertical;
    PCrop.rightMargin = padAmountHorizontal;
    PCrop.bottomMargin = padAmountVertical;
    PCrop.mode = Crop.prototype.AbsolutePixels;
    PCrop.xResolution = 72.000;
    PCrop.yResolution = 72.000;
    PCrop.metric = false;
    PCrop.forceResolution = false;
    PCrop.red = 0.000000;
    PCrop.green = 0.000000;
    PCrop.blue = 0.000000;
    PCrop.alpha = 1.000000;
    PCrop.noGUIMessages = false;

    PCrop.executeOn(paddedImageView);

    Console.warningln('Padded image width: ' + paddedImageView.image.width + ' and height: ' + paddedImageView.image.height);

    return paddedImageView;
}


function padAndFillImage(view, padAmountHorizontal, padAmountVertical, paddingFillType) {
    // Console.warningln('padAmountHorizontal: ' + padAmountHorizontal + ' padAmountVertical: ' + padAmountVertical);

    let newPaddedImageView = padImage(view, padAmountHorizontal, padAmountVertical);
    // let newPaddedImageView = padImage(view, 0, 0);

    // return newPaddedImageView;

    var PCrop = new Crop;

    PCrop.mode = Crop.prototype.AbsolutePixels;
    PCrop.xResolution = 72.000;
    PCrop.yResolution = 72.000;
    PCrop.metric = false;
    PCrop.forceResolution = false;
    PCrop.red = 0.000000;
    PCrop.green = 0.000000;
    PCrop.blue = 0.000000;
    PCrop.alpha = 1.000000;
    PCrop.noGUIMessages = false;

    let paddedImageWindow = new ImageWindow(view.image.width, view.image.height,
        view.image.numberOfChannels, // 1 channel for grayscale
        view.image.bitsPerSample,
        view.image.isReal,
        view.image.isColor
    );

    let paddedImageView = paddedImageWindow.mainView;
    paddedImageView.beginProcess(UndoFlag_NoSwapFile);
    let paddedImage = paddedImageView.image;
    paddedImage.apply(view.image);

    paddedImageWindow.show();

    // Fill by specified algorithm
    switch (paddingFillType) {
        case PADDING_SYMMETRIC:
            if (padAmountHorizontal > 0) {
                // Horizontal pad left

                PCrop.leftMargin = 0;
                PCrop.topMargin = 0;
                PCrop.rightMargin = padAmountHorizontal - paddedImageView.image.width;
                PCrop.bottomMargin = 0;

                PCrop.executeOn(paddedImageView);

                Console.warningln('Left padding image width: ' + paddedImageView.image.width + ' and height: ' + paddedImageView.image.height);

                paddedImageView.image = paddedImageView.image.mirrorHorizontal();

                newPaddedImageView.image.apply(paddedImageView.image, ImageOp_Screen, new Point(0, padAmountVertical));

                paddedImage.apply(view.image);
                processEvents();

                paddedImageWindow.forceClose();

                // if the original image width is odd, make it even by adding 1 more pixel to the right margin
                /*
                if (view.image.width % 2) {
                    padAmountHorizontal += 1;
                }
                */
                paddedImageWindow = new ImageWindow(view.image.width, view.image.height,
                    view.image.numberOfChannels, // 1 channel for grayscale
                    view.image.bitsPerSample,
                    view.image.isReal,
                    view.image.isColor
                );

                paddedImageView = paddedImageWindow.mainView;
                paddedImageView.beginProcess(UndoFlag_NoSwapFile);
                paddedImage = paddedImageView.image;
                paddedImage.apply(view.image);

                paddedImageWindow.show();

                // Horizontal pad right
                let rightPadWidth = padAmountHorizontal - view.image.width;
                Console.warningln('rightPadWidth: ' + rightPadWidth);

                PCrop.leftMargin = padAmountHorizontal - view.image.width;
                PCrop.topMargin = 0;
                PCrop.rightMargin = 0;
                PCrop.bottomMargin = 0;

                PCrop.executeOn(paddedImageView);

                Console.warningln('Right padding image width: ' + paddedImageView.image.width + ' and height: ' + paddedImageView.image.height);

                paddedImageView.image = paddedImageView.image.mirrorHorizontal();

                newPaddedImageView.image.apply(paddedImageView.image, ImageOp_Screen, new Point(padAmountHorizontal + view.image.width, padAmountVertical));

                paddedImage.apply(view.image);

                paddedImageWindow.forceClose();
                paddedImageWindow = new ImageWindow(view.image.width, view.image.height,
                    view.image.numberOfChannels, // 1 channel for grayscale
                    view.image.bitsPerSample,
                    view.image.isReal,
                    view.image.isColor
                );

                paddedImageView = paddedImageWindow.mainView;
                paddedImageView.beginProcess(UndoFlag_NoSwapFile);
                paddedImage = paddedImageView.image;
                paddedImage.apply(view.image);

                paddedImageWindow.show();
            }

            if (padAmountVertical > 0) {
                // Vertical pad top
                PCrop.leftMargin = 0;
                PCrop.topMargin = 0;
                PCrop.rightMargin = 0;
                PCrop.bottomMargin = padAmountVertical - view.image.height;

                PCrop.executeOn(paddedImageView);

                paddedImageView.image = paddedImageView.image.mirrorVertical();

                newPaddedImageView.image.apply(paddedImageView.image, ImageOp_Screen, new Point(padAmountHorizontal, 0));

                paddedImage.apply(view.image);

                paddedImageWindow.show();

                paddedImageWindow.forceClose();

                // if the original image height is odd, make it even by adding 1 more pixel to the bottom margin
                /*
                if (view.image.height % 2) {
                    padAmountVertical += 1;
                }
                */
                paddedImageWindow = new ImageWindow(view.image.width, view.image.height,
                    view.image.numberOfChannels, // 1 channel for grayscale
                    view.image.bitsPerSample,
                    view.image.isReal,
                    view.image.isColor
                );

                paddedImageView = paddedImageWindow.mainView;
                paddedImageView.beginProcess(UndoFlag_NoSwapFile);
                paddedImage = paddedImageView.image;
                paddedImage.apply(view.image);

                paddedImageWindow.show();

                // Horizontal pad bottom
                PCrop.leftMargin = 0;
                PCrop.topMargin = padAmountVertical - view.image.height;
                PCrop.rightMargin = 0;
                PCrop.bottomMargin = 0;

                PCrop.executeOn(paddedImageView);

                paddedImageView.image.mirrorVertical();

                newPaddedImageView.image.apply(paddedImageView.image, ImageOp_Screen, new Point(padAmountHorizontal, padAmountVertical + view.image.height));
            }

            paddedImageWindow.forceClose();

            break;
        case PADDING_ZERO:
            // Crop above has already zero padded
            paddedImageWindow.forceClose();
            break;
        case PADDING_REFLECT:
            // Code to execute if expression === value2
            break;
        case PADDING_CONSTANT:
            // Code to execute if expression === value2
            break;
        default:
        // Code to execute if expression does not match any case
    }

    // Console.warningln('Final padded image width: ' + newPaddedImageView.image.width + ' height: ' + newPaddedImageView.image.height);

    return newPaddedImageView;
}


function convolve2D(view, kernel, axis, scaleConvolution) {
    inputImage = view.image;

    const inputRows = inputImage.height;
    const inputCols = inputImage.width;
    Console.warningln('inputRows: ' + inputRows + ' inputCols:' + inputCols);
    const kernelLength = kernel.length;
    Console.warningln('kernelLength: ' + kernelLength + "kernel: " + kernel);
    const kernelOffset = Math.floor(kernelLength / 2);
    let convolutionScaleFactor = 1;

    // Console.warningln('kernel: ' + kernel);
    // return;

    if (scaleConvolution) {
        let kernelSum = kernel.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
        convolutionScaleFactor = convolutionScaleFactor * kernelSum;
    }
    // let kernelSum = kernel.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
    // const kernelCols = kernel[0].length;

    const outputRows = inputRows;
    const outputCols = inputCols;
    /*
    if (axis === HORIZONTAL_AXIS) {
        outputRows = inputRows;
        outputCols = inputCols;
    }
    else {
        outputRows = inputRows - kernelRows + 1;
        outputCols = inputCols - kernelCols + 1;
    }
    */
    // const outputMatrix = Array(outputRows).fill(null).map(() => Array(outputCols).fill(0));
    /*
    let outputMatrix = [];

    for (let i = 0; i < outputRows; i++) {
        outputMatrix[i] = [];
        for (let j = 0; j < outputCols; j++) {
            outputMatrix[i][j] = 0; // Initialize with default value of 0
        }
    }
    */
    let outputImageWindow;

    let subtractionFactor = 1;

    if (kernelOffset % 2) {
        subtractionFactor = subtractionFactor + 1;
    }

    let padAmount = (kernelOffset - 1) * 2;

    if (axis === HORIZONTAL_AXIS) {
        // outputImageWindow = new ImageWindow(Math.ceil((view.image.width - kernelOffset * 2 + 1) / 2), view.image.height,
        let originalImageWidthPrePadding = view.image.width - 2 * padAmount;
        let desiredHalfWidth = Math.round(originalImageWidthPrePadding / 2) + kernelOffset - 1;
        Console.warningln('desiredHalfWidth: ' + desiredHalfWidth + ' originalImageWidthPrePadding:' + originalImageWidthPrePadding);
        outputImageWindow = new ImageWindow(desiredHalfWidth, view.image.height,
            view.image.numberOfChannels, // 1 channel for grayscale
            view.image.bitsPerSample,
            view.image.isReal,
            view.image.isColor
        );
    }
    else {
        // outputImageWindow = new ImageWindow(view.image.width, Math.ceil((view.image.height - kernelOffset * 2 + 1) / 2),
        let originalImageHeightPrePadding = view.image.height - 2 * padAmount;
        let desiredHalfHeight = Math.round(originalImageHeightPrePadding / 2) + kernelOffset - 1;
        Console.warningln('desiredHalfHeight: ' + desiredHalfHeight + ' originalImageHeightPrePadding:' + originalImageHeightPrePadding);
        // outputImageWindow = new ImageWindow(view.image.width, Math.round((view.image.height / 2) - subtractionFactor),
        outputImageWindow = new ImageWindow(view.image.width, desiredHalfHeight,
            view.image.numberOfChannels, // 1 channel for grayscale
            view.image.bitsPerSample,
            view.image.isReal,
            view.image.isColor
        );
    }

    // return;

    let outputImageView = outputImageWindow.mainView;
    outputImageView.beginProcess(UndoFlag_NoSwapFile);
    let outputImage = outputImageView.image;
    outputImage.apply(0);

    outputImageWindow.show();

    // return;

    let convSubtractionFactor = 2;

    if (kernelOffset % 2) {
        convSubtractionFactor = 4;
    }

    Console.warningln('subtractionFactor: ' + subtractionFactor + ' convSubtractionFactor:' + convSubtractionFactor);

    // Console.warningln('outputRows: ' + outputMatrix.length + ' outputCols:' + outputMatrix[0].length);
    Console.warningln('outputRows: ' + outputImage.height + ' outputCols:' + outputImage.width);

    if (axis === HORIZONTAL_AXIS) {
        Console.warningln('Horizontal');

        for (let i = 0; i < outputRows; i++) {
            for (let j = 0; j < outputImage.width * 2; j += 2) {
                let sum = 0;
                for (let m = 0; m < kernelLength; m++) {
                    if (j + m < inputImage.width) {
                        // Console.warningln('j: ' + j + ' m:' + m);
                        sum += inputImage.sample(j + m, i) * kernel[m];
                    }
                }
                outputImage.setSample(sum / convolutionScaleFactor, j / 2, i);
            }
        }

        // return;
    }
    else {
        Console.warningln('Vertical');

        for (let i = 0; i < outputImage.height * 2; i += 2) {
            for (let j = 0; j < outputCols; j++) {
                let sum = 0;
                for (let m = 0; m < kernelLength; m++) {
                    sum += inputImage.sample(j, i + m) * kernel[m];
                }
                // outputMatrix[i][j] = sum;
                outputImage.setSample(sum / convolutionScaleFactor, j, i / 2);
            }
        }
    }
    /*
    for (let i = 0; i < outputRows; i++) {
        for (let j = 0; j < outputCols; j++) {
            inputImage.setSample(outputMatrix[i][j], j, i);
        }
    }
    */

    processEvents();

    return outputImageView;
}


function convolve2DReverseScaling(view, linearKernel, axis, scaleConvolution) {
    let localScaleFactor = 1;

    if (scaleConvolution === false) {
        localScaleFactor = 10000;
    }

    // Create a scaled kernel array
    let kernelSum = 0;
    for (let k = 0; k < linearKernel.length; k++) {
        linearKernel[k] = Math.floor(linearKernel[k] * localScaleFactor);

        kernelSum = kernelSum + linearKernel[k];
    }

    Console.warningln('kernelSum: ' + kernelSum);

    let kernelLength = linearKernel.length;

    let kernelPadding = Math.floor(kernelLength / 2);

    let kernelArray = [];

    for (let i = 0; i < kernelLength; i++) {
        kernelArray[i] = [];
        for (let j = 0; j < kernelLength; j++) {
            kernelArray[i][j] = 0; // Initialize with default value of 0
        }
    }

    var PMath = new PixelMath;
    PMath.expression = view.id + " / 100000000";
    PMath.expression1 = "";
    PMath.expression2 = "";
    PMath.expression3 = "";
    PMath.useSingleExpression = true;
    PMath.symbols = "";
    PMath.clearImageCacheAndExit = false;
    PMath.cacheGeneratedImages = false;
    PMath.generateOutput = true;
    PMath.singleThreaded = false;
    PMath.optimization = true;
    PMath.use64BitWorkingImage = false;
    PMath.rescale = false;
    PMath.rescaleLower = 0;
    PMath.rescaleUpper = 1;
    PMath.truncate = false;
    PMath.truncateLower = 0;
    PMath.truncateUpper = 1;
    PMath.createNewImage = false;
    PMath.showNewImage = true;
    PMath.newImageId = "";
    PMath.newImageWidth = 0;
    PMath.newImageHeight = 0;
    PMath.newImageAlpha = false;
    PMath.newImageColorSpace = PixelMath.prototype.SameAsTarget;
    PMath.newImageSampleFormat = PixelMath.prototype.SameAsTarget;
    /*
     * Read-only properties
     *
    P.outputData = [ // globalVariableId, globalVariableRK, globalVariableG, globalVariableB
    ];
     */

    var PResample = new Resample;
    PResample.xSize = 1.0000;
    PResample.ySize = 0.5000;
    PResample.mode = Resample.prototype.RelativeDimensions;
    PResample.absoluteMode = Resample.prototype.ForceWidthAndHeight;
    PResample.xResolution = 72.000;
    PResample.yResolution = 72.000;
    PResample.metric = false;
    PResample.forceResolution = false;
    PResample.interpolation = Resample.prototype.Auto;
    PResample.clampingThreshold = 0.30;
    PResample.smoothness = 1.50;
    PResample.gammaCorrection = false;
    PResample.noGUIMessages = false;


    if (axis === VERTICAL_AXIS) {
        // Vertical convolution
        for (let kernelElement = 0; kernelElement < kernelLength; kernelElement++) {
            kernelArray[kernelElement][kernelPadding] = linearKernel[kernelElement];
        }

        // Now flatten the array
        let flattenedKernelArray = [];
        for (let i = 0; i < kernelLength; i++) {
            for (let j = 0; j < kernelLength; j++) {
                flattenedKernelArray.push(kernelArray[i][j]);
            }
        }

        // Add a row if the height is an odd number
        if (view.image.height % 2) {
            var PCrop = new Crop;

            PCrop.mode = Crop.prototype.AbsolutePixels;
            PCrop.xResolution = 72.000;
            PCrop.yResolution = 72.000;
            PCrop.metric = false;
            PCrop.forceResolution = false;
            PCrop.red = 0.000000;
            PCrop.green = 0.000000;
            PCrop.blue = 0.000000;
            PCrop.alpha = 1.000000;
            PCrop.noGUIMessages = false;

            PCrop.leftMargin = 0;
            PCrop.topMargin = 0;
            PCrop.rightMargin = 0;
            PCrop.bottomMargin = 1;

            PCrop.executeOn(view);

            for (let col = 0; col < view.image.width; col++) {
                view.image.setSample(view.image.sample(col, view.image.height - 2), col, view.image.height - 1);
            }
        }

        Console.warningln('flattenedKernelArray: ' + flattenedKernelArray);

        view.image.convolve(flattenedKernelArray, 2);

        if (!scaleConvolution) {
            // Now divide by the scale factor
            // PMath.executeOn(view);
        }

        // Now resample by a factor of 2 along the vertical axis
        PResample.xSize = 1.0000;
        PResample.ySize = 0.5000;

        // PResample.executeOn(view);

        let downsampledView = downsampleImage2X(view, axis);
        view.window.forceClose();

        view = downsampledView;
    }
    else {
        // Horizontal convolution
        for (let kernelElement = 0; kernelElement < kernelLength; kernelElement++) {
            kernelArray[kernelPadding][kernelElement] = linearKernel[kernelElement];
        }

        // Now flatten the array
        let flattenedKernelArray = [];
        for (let i = 0; i < kernelLength; i++) {
            for (let j = 0; j < kernelLength; j++) {
                flattenedKernelArray.push(kernelArray[i][j]);
            }
        }

        // Add a column if the width is an odd number
        if (view.image.width % 2) {
            var PCrop = new Crop;

            PCrop.mode = Crop.prototype.AbsolutePixels;
            PCrop.xResolution = 72.000;
            PCrop.yResolution = 72.000;
            PCrop.metric = false;
            PCrop.forceResolution = false;
            PCrop.red = 0.000000;
            PCrop.green = 0.000000;
            PCrop.blue = 0.000000;
            PCrop.alpha = 1.000000;
            PCrop.noGUIMessages = false;

            PCrop.leftMargin = 0;
            PCrop.topMargin = 0;
            PCrop.rightMargin = 1;
            PCrop.bottomMargin = 0;

            PCrop.executeOn(view);

            for (let row = 0; row < view.image.height; row++) {
                view.image.setSample(view.image.sample(view.image.width - 2, row), view.image.width - 1, row);
            }
        }

        Console.warningln('flattenedKernelArray: ' + flattenedKernelArray);

        view.image.convolve(flattenedKernelArray, 2);

        if (!scaleConvolution) {
            // Now divide by the scale factor
            // PMath.executeOn(view);
        }

        // Now resample by a factor of 2 along the horizontal axis
        PResample.xSize = 0.5000;
        PResample.ySize = 1.0000;

        // PResample.executeOn(view);

        let downsampledView = downsampleImage2X(view, axis);
        view.window.forceClose();

        view = downsampledView;
    }

    return view;
}


function idwt2Javascript(coeffs, wavelet) {
    const { ca, ch, cv, cd } = coeffs;
    const filterLength = wavelet.length;

    function convolveAndUpsample(signal, filter) {
        const upsampled = [];
        for (let i = 0; i < signal.length; i++) {
            upsampled.push(signal[i]);
            upsampled.push(0);
        }

        const padded = Array(filterLength - 1).fill(0).concat(upsampled).concat(Array(filterLength - 1).fill(0));
        const convolved = [];
        for (let i = 0; i <= padded.length - filterLength; i++) {
            let sum = 0;
            for (let j = 0; j < filterLength; j++) {
                sum += padded[i + j] * filter[j];
            }
            convolved.push(sum);
        }
        return convolved;
    }


    function transpose(matrix) {
        return matrix[0].map((col, i) => matrix.map(row => row[i]));
    }

    function idwt1D(cA, cD, w) {
        const rec_l = convolveAndUpsample(cA, w.slice().reverse());
        const rec_h = convolveAndUpsample(cD, w.slice().reverse());

        const result = [];
        for (let i = 0; i < Math.max(rec_l.length, rec_h.length); i++) {
            const val_l = rec_l[i] || 0;
            const val_h = rec_h[i] || 0;
            result.push(val_l + val_h);
        }
        return result.filter((_, index) => index % 2 === 0);
    }

    function processRows(matrix, w) {
        return matrix.map(row => idwt1D(row.cA, row.cD, w));
    }

    function processColumns(matrix, w) {
        const transposedMatrix = transpose(matrix);
        const result = processRows(transposedMatrix.map(col => ({ cA: col.slice(0, col.length / 2), cD: col.slice(col.length / 2) })), w);
        return transpose(result);
    }

    const w_r = wavelet.slice().reverse();

    // Inverse DWT along rows
    const intermediate_rows = processRows([
        { cA: ca[0], cD: ch[0] },
        { cA: cv[0], cD: cd[0] }
    ], w_r);

    // Inverse DWT along columns
    const result_2d = processColumns(intermediate_rows.map(row => [row.slice(0, row.length / 2), row.slice(row.length / 2)]), w_r);
    return result_2d;
}

function idwt_buffer_length(coeffs_len, filter_len) {
    
    return 2 * coeffs_len - filter_len + 2;
}


function idwt2DPython() {
    /*
    import numpy as np

    def idwt2_daubechies(coeffs, wavelet = 'db4'):
    """
    Performs a single - level 2D inverse discrete wavelet transform using Daubechies wavelets.

        Args:
            coeffs (tuple): A tuple containing the approximation coefficients(cA) and detail coefficients(cH, cV, cD).
                wavelet(str, optional): The Daubechies wavelet to use(e.g., 'db4', 'db6').Defaults to 'db4'.

                    Returns:
    numpy.ndarray: The reconstructed 2D array.
    """

    if wavelet == 'db4':
        h = [0.482962913145, 0.836516303738, 0.224143868042, -0.129409522585]
    g = [-0.129409522585, -0.224143868042, 0.836516303738, -0.482962913145]
    elif wavelet == 'db6':
    h = [0.332670552950, 0.806891509311, 0.459877502118, -0.135287178931, -0.085441273882, 0.035226291882]
    g = [0.035226291882, 0.085441273882, -0.135287178931, -0.459877502118, 0.806891509311, -0.332670552950]
    else:
        raise ValueError("Unsupported Daubechies wavelet.")

    cA, (cH, cV, cD) = coeffs
    
    def idwt_step(a, d, w):
    """Performs a 1D inverse DWT step."""
    L = len(w)
    N = (len(a) + len(d))
    y = np.zeros(N)

    for i in range(len(a)):
        for k in range(L):
            y[2 * i + k] += a[i] * w[k]
    for i in range(len(d)):
        for k in range(L):
            y[2 * i + k] += d[i] * g[k]
    return y

    # Inverse DWT along rows
    rows = cA.shape[0]
    cols = cA.shape[1]

    irows = np.zeros((rows, cols * 2))
    for i in range(rows):
        irows[i, :] = idwt_step(cA[i, :], cH[i, :], h)

    # Inverse DWT along columns
    result = np.zeros((rows * 2, cols * 2))
    for j in range(cols * 2):
        result[:, j] = idwt_step(irows[:, j], cV[:, j], h)

    return result
    */
}

function downsampleImage2X(view, axis) {
    // Save the even numbered samples and discard the odd numbered samples as per Matlab
    let originalWidth = view.image.width;
    let originalHeight = view.image.height;

    if (axis === VERTICAL_AXIS) {
        let downsampledImageWindow = new ImageWindow(originalWidth / 2, originalHeight,
            view.image.numberOfChannels, // 1 channel for grayscale
            view.image.bitsPerSample,
            view.image.isReal,
            view.image.isColor
        );

        let downsampledImageView = downsampledImageWindow.mainView;
        downsampledImageView.beginProcess(UndoFlag_NoSwapFile);
        let downsampledImage = downsampledImageView.image;
        downsampledImage.apply(0);

        downsampledImageWindow.show();

        for (let i = 0; i < originalHeight; i++) {
            for (let j = 0; j < originalWidth / 2; j += 2) {
                downsampledImage.setSample(view.image.sample(j * 2, i), j, i);
            }
        }

        return downsampledImageView;
    }
    else {
        let downsampledImageWindow = new ImageWindow(originalWidth, originalHeight / 2,
            view.image.numberOfChannels, // 1 channel for grayscale
            view.image.bitsPerSample,
            view.image.isReal,
            view.image.isColor
        );

        let downsampledImageView = downsampledImageWindow.mainView;
        downsampledImageView.beginProcess(UndoFlag_NoSwapFile);
        let downsampledImage = downsampledImageView.image;
        downsampledImage.apply(0);

        downsampledImageWindow.show();

        for (let i = 0; i < originalHeight / 2; i += 2) {
            for (let j = 0; j < originalWidth; j++) {
                downsampledImage.setSample(view.image.sample(j, i * 2), j, i);
            }
        }

        return downsampledImageView;
    }

    // should never get to this point, but return view anyways
    return view;
}

function upsampleAndConvolveImage2X(CA, CD, axis, lowPassRecKernel, highPassRecKernel, paddingFillType) {
    // Save the even numbered samples and discard the odd numbered samples as per Matlab
    let originalWidth = CA.image.width;
    let originalHeight = CA.image.height;

    if (axis === VERTICAL_AXIS) {
        let CAupsampledImageWindow = new ImageWindow(originalWidth, originalHeight * 2,
            CA.image.numberOfChannels, // 1 channel for grayscale
            CA.image.bitsPerSample,
            CA.image.isReal,
            CA.image.isColor
        );

        let CAupsampledImageView = CAupsampledImageWindow.mainView;
        CAupsampledImageView.beginProcess(UndoFlag_NoSwapFile);
        let CAupsampledImage = CAupsampledImageView.image;
        CAupsampledImage.apply(0);

        CAupsampledImageWindow.show();

        Console.warningln('originalWidth: ' + originalWidth + " originalHeight: " + originalHeight);
        
        for (let i = 0; i < originalHeight; i++) {
            for (let j = 0; j < originalWidth; j++) {
                CAupsampledImage.setSample(CA.image.sample(j, i), j, i * 2);
            }
        }
        
        let CDupsampledImageWindow = new ImageWindow(originalWidth, originalHeight * 2,
            CA.image.numberOfChannels, // 1 channel for grayscale
            CA.image.bitsPerSample,
            CA.image.isReal,
            CA.image.isColor
        );

        let CDupsampledImageView = CDupsampledImageWindow.mainView;
        CDupsampledImageView.beginProcess(UndoFlag_NoSwapFile);
        let CDupsampledImage = CDupsampledImageView.image;
        CDupsampledImage.apply(0);

        CDupsampledImageWindow.show();
        
        for (let i = 0; i < originalHeight; i++) {
            for (let j = 0; j < originalWidth; j++) {
                CDupsampledImage.setSample(CD.image.sample(j, i), j, i * 2);
            }
        }

        // return;
        
        // Kernel calculations
        const kernelLength = lowPassRecKernel.length;
        const kernelOffset = Math.floor(kernelLength / 2);
        let convolutionScaleFactor = 1;

        let kernelSum = lowPassRecKernel.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
        convolutionScaleFactor = convolutionScaleFactor * kernelSum;

        // Now pad top and bottom by half the filter size
        let padAmountHorizontal = 0;
        let padAmountVertical = kernelOffset;
        // let paddingFillType = PADDING_SYMMETRIC;

        let CAPaddedAndFilledImageView = padAndFillImage(CAupsampledImageView, padAmountHorizontal, padAmountVertical, paddingFillType);
        let CDPaddedAndFilledImageView = padAndFillImage(CDupsampledImageView, padAmountHorizontal, padAmountVertical, paddingFillType);

        // return;

        // Now convolve the reconstruction filter down the new padded and filled and upsampled image
        let outputImageWindow;

        outputImageWindow = new ImageWindow(CAPaddedAndFilledImageView.image.width, CAPaddedAndFilledImageView.image.height,
            CAPaddedAndFilledImageView.image.numberOfChannels, // 1 channel for grayscale
            CAPaddedAndFilledImageView.image.bitsPerSample,
            CAPaddedAndFilledImageView.image.isReal,
            CAPaddedAndFilledImageView.image.isColor
        );

        let outputImageView = outputImageWindow.mainView;
        outputImageView.beginProcess(UndoFlag_NoSwapFile);
        let outputImage = outputImageView.image;
        outputImage.apply(0);

        outputImageWindow.show();

        for (let i = 0; i < CAPaddedAndFilledImageView.image.height - kernelLength + 1; i++) {
            for (let j = 0; j < CAPaddedAndFilledImageView.image.width; j++) {
                let sum = 0;
                for (let m = 0; m < kernelLength; m++) {
                    sum += (CAPaddedAndFilledImageView.image.sample(j, i + m) * lowPassRecKernel[m]) * convolutionScaleFactor;
                    sum += CDPaddedAndFilledImageView.image.sample(j, i + m) * highPassRecKernel[m];
                }
                // outputMatrix[i][j] = sum;
                outputImage.setSample(sum, j, i);
                // outputImage.setSample(CA.image.sample(j, i), j, i * 2);
            }
        }
        /*
        for (; i < N; ++i, o += 2) {
        size_t j;
            for (j = 0; j < F / 2; ++j) {
                output[o] += filter[j * 2] * input[i - j];
                output[o + 1] += filter[j * 2 + 1] * input[i - j];
            }
        }
        */

        CAupsampledImageWindow.forceClose();
        CDupsampledImageWindow.forceClose();

        CAPaddedAndFilledImageView.window.forceClose();
        CDPaddedAndFilledImageView.window.forceClose();

        // return;

        return outputImageView;
    }
    else {
        let CAupsampledImageWindow = new ImageWindow(originalWidth * 2, originalHeight,
            CA.image.numberOfChannels, // 1 channel for grayscale
            CA.image.bitsPerSample,
            CA.image.isReal,
            CA.image.isColor
        );

        Console.warningln('originalWidth: ' + originalWidth + " originalHeight: " + originalHeight);

        let CAupsampledImageView = CAupsampledImageWindow.mainView;
        CAupsampledImageView.beginProcess(UndoFlag_NoSwapFile);
        let CAupsampledImage = CAupsampledImageView.image;
        CAupsampledImage.apply(0);

        CAupsampledImageWindow.show();
        
        for (let i = 0; i < originalHeight; i++) {
            for (let j = 0; j < originalWidth; j++) {
                CAupsampledImage.setSample(CA.image.sample(j, i), j * 2, i);
            }
        }
        
        let CDupsampledImageWindow = new ImageWindow(originalWidth * 2, originalHeight,
            CA.image.numberOfChannels, // 1 channel for grayscale
            CA.image.bitsPerSample,
            CA.image.isReal,
            CA.image.isColor
        );

        let CDupsampledImageView = CDupsampledImageWindow.mainView;
        CDupsampledImageView.beginProcess(UndoFlag_NoSwapFile);
        let CDupsampledImage = CDupsampledImageView.image;
        CDupsampledImage.apply(0);

        CDupsampledImageWindow.show();
        
        for (let i = 0; i < originalHeight; i++) {
            for (let j = 0; j < originalWidth; j++) {
                CDupsampledImage.setSample(CD.image.sample(j, i), j * 2, i);
            }
        }

        // return;
        
        // Kernel calculations
        let kernelLength = lowPassRecKernel.length;
        let kernelOffset = Math.floor(kernelLength / 2);
        let convolutionScaleFactor = 1;

        let kernelSum = lowPassRecKernel.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
        convolutionScaleFactor = convolutionScaleFactor * kernelSum;

        // Console.warningln('kernelSum: ' + kernelSum);
        // Console.warningln('convolutionScaleFactor: ' + convolutionScaleFactor);

        // return;

        // Now pad top and bottom by half the filter size
        let padAmountHorizontal = kernelOffset;
        let padAmountVertical = 0;
        // let paddingFillType = PADDING_SYMMETRIC;

        let CAPaddedAndFilledImageView = padAndFillImage(CAupsampledImageView, padAmountHorizontal, padAmountVertical, paddingFillType);
        let CDPaddedAndFilledImageView = padAndFillImage(CDupsampledImageView, padAmountHorizontal, padAmountVertical, paddingFillType);

        // return;

        // Now convolve the reconstruction filter down the new padded and filled and upsampled image
        let outputImageWindow;

        outputImageWindow = new ImageWindow(CAPaddedAndFilledImageView.image.width, CAPaddedAndFilledImageView.image.height,
            CAPaddedAndFilledImageView.image.numberOfChannels, // 1 channel for grayscale
            CAPaddedAndFilledImageView.image.bitsPerSample,
            CAPaddedAndFilledImageView.image.isReal,
            CAPaddedAndFilledImageView.image.isColor
        );

        let outputImageView = outputImageWindow.mainView;
        outputImageView.beginProcess(UndoFlag_NoSwapFile);
        let outputImage = outputImageView.image;
        outputImage.apply(0);

        outputImageWindow.show();

        for (let i = 0; i < CAPaddedAndFilledImageView.image.height; i++) {
            for (let j = 0; j < CAPaddedAndFilledImageView.image.width - kernelLength + 1; j++) {
                let sum = 0;
                for (let m = 0; m < kernelLength; m++) {
                    sum += (CAPaddedAndFilledImageView.image.sample(j + m, i) * lowPassRecKernel[m]) * convolutionScaleFactor;
                    sum += CDPaddedAndFilledImageView.image.sample(j + m, i) * highPassRecKernel[m];
                }
                // outputMatrix[i][j] = sum;
                outputImage.setSample(sum, j, i);
                // outputImage.setSample(CA.image.sample(j, i), j * 2, i);
                /*
                Console.warningln('lowPassRecKernel: ' + lowPassRecKernel + ' highPassRecKernel:' + highPassRecKernel);
                Console.warningln('CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 0, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 1, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 2, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 3, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 4, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 5, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 6, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 7, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 8, i) +
                    'CA 0,0: ' + CAPaddedAndFilledImageView.image.sample(j + 9, i));
                Console.warningln('CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 0, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 1, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 2, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 3, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 4, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 5, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 6, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 7, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 8, i) +
                    'CA 0,0: ' + CDPaddedAndFilledImageView.image.sample(j + 9, i));

                return outputImageView;
                */
            }
        }
        // return;
        CAupsampledImageWindow.forceClose();
        CDupsampledImageWindow.forceClose();

        CAPaddedAndFilledImageView.window.forceClose();
        CDPaddedAndFilledImageView.window.forceClose();

        // return;

        return outputImageView;
    }

    // should never get to this point, but return view anyways
    return CA;
}

function idwt(CA, CD, kernel, axis, scaleConvolution) {
    let baseWidth = LL.image.width;
    let baseHeight = LL.image.height;


}

function idwt2D(LL, HL, LH, HH, waveletType) {
    let baseWidth = LL.image.width;
    let baseHeight = LL.image.height;


}

function idwt2(LL, HL, LH, HH, waveletType) {
    let wavelet = new Wavelet(waveletType);

    let filterType = 'rec_lo';
    const lowSynthesis = wavelet.getKernel(filterType);
    filterType = 'rec_hi';
    const highSynthesis = wavelet.getKernel(filterType);

    const filterLength = lowSynthesis.length;

    const reconstructedWidth = (LL.image.width - 1) * 2 + filterLength;
    const reconstructedHeight = (LL.image.height - 1) * 2 + filterLength;

    // const reconstructed = Array(reconstructedHeight).fill(null).map(() => Array(reconstructedWidth).fill(0));

    let paddedImageWindow = new ImageWindow(reconstructedWidth, reconstructedHeight,
        LL.image.numberOfChannels, // 1 channel for grayscale
        LL.image.bitsPerSample,
        LL.image.isReal,
        LL.image.isColor
    );

    let paddedImageView = paddedImageWindow.mainView;
    paddedImageView.beginProcess(UndoFlag_NoSwapFile);
    let paddedImage = paddedImageView.image;
    paddedImage.apply(view.image);

    paddedImageWindow.show();


    // Synthesis filters
    // const lowSynthesis = wavelet.rec_lo;
    // const highSynthesis = wavelet.rec_hi;

    // Helper function for 1D IDWT
    function idwt1(cA, cD, synthesisLow, synthesisHigh) {
        const resultLength = (cA.length - 1) * 2 + synthesisLow.length;
        const result = Array(resultLength).fill(0);

        for (let i = 0; i < cA.length; i++) {
            for (let j = 0; j < synthesisLow.length; j++) {
                result[i * 2 + j] += cA[i] * synthesisLow[j];
            }
        }

        for (let i = 0; i < cD.length; i++) {
            for (let j = 0; j < synthesisHigh.length; j++) {
                result[i * 2 + j] += cD[i] * synthesisHigh[j];
            }
        }
        return result;
    }

    // Reconstruct rows
    const rowReconstructedLL = LL.map(row => idwt1(row, Array(row.length).fill(0), lowSynthesis, highSynthesis));
    const rowReconstructedHL = HL.map(row => idwt1(row, Array(row.length).fill(0), lowSynthesis, highSynthesis));
    const rowReconstructedLH = LH.map(row => idwt1(row, Array(row.length).fill(0), lowSynthesis, highSynthesis));
    const rowReconstructedHH = HH.map(row => idwt1(row, Array(row.length).fill(0), lowSynthesis, highSynthesis));

    // Reconstruct columns
    for (let col = 0; col < reconstructedWidth; col++) {
        const colLL = rowReconstructedLL.map(row => row[col]);
        const colHL = rowReconstructedHL.map(row => row[col]);
        const colLH = rowReconstructedLH.map(row => row[col]);
        const colHH = rowReconstructedHH.map(row => row[col]);

        const reconstructedCol = idwt1(colLL, colLH, lowSynthesis, highSynthesis);

        for (let row = 0; row < reconstructedHeight; row++) {
            reconstructed[row][col] = reconstructedCol[row];
        }
    }


    return reconstructed;
}

function dwt(view, padAmountHorizontal, padAmountVertical, paddingFillType, waveletType, filterType, axis, scaleConvolution) {
    if (axis === VERTICAL_AXIS) {
        // Add a row if the height is an odd number
        if (view.image.height % 2) {
            Console.warningln('Adding vertical row to odd number of rows');
            var PCrop = new Crop;

            PCrop.mode = Crop.prototype.AbsolutePixels;
            PCrop.xResolution = 72.000;
            PCrop.yResolution = 72.000;
            PCrop.metric = false;
            PCrop.forceResolution = false;
            PCrop.red = 0.000000;
            PCrop.green = 0.000000;
            PCrop.blue = 0.000000;
            PCrop.alpha = 1.000000;
            PCrop.noGUIMessages = false;

            PCrop.leftMargin = 0;
            PCrop.topMargin = 0;
            PCrop.rightMargin = 0;
            PCrop.bottomMargin = 1;

            PCrop.executeOn(view);

            for (let col = 0; col < view.image.width; col++) {
                view.image.setSample(view.image.sample(col, view.image.height - 2), col, view.image.height - 1);
            }
        }
    }
    else {
        // Add a column if the width is an odd number
        Console.warningln('Adding horizontal column to odd number of columns');
        if (view.image.width % 2) {
            var PCrop = new Crop;

            PCrop.mode = Crop.prototype.AbsolutePixels;
            PCrop.xResolution = 72.000;
            PCrop.yResolution = 72.000;
            PCrop.metric = false;
            PCrop.forceResolution = false;
            PCrop.red = 0.000000;
            PCrop.green = 0.000000;
            PCrop.blue = 0.000000;
            PCrop.alpha = 1.000000;
            PCrop.noGUIMessages = false;

            PCrop.leftMargin = 0;
            PCrop.topMargin = 0;
            PCrop.rightMargin = 1;
            PCrop.bottomMargin = 0;

            PCrop.executeOn(view);

            for (let row = 0; row < view.image.height; row++) {
                view.image.setSample(view.image.sample(view.image.width - 2, row), view.image.width - 1, row);
            }
        }
    }

    let newPaddedAndFilledImageView = padAndFillImage(view, padAmountHorizontal, padAmountVertical, paddingFillType);

    // return;

    Console.warningln('New padded width: ' + newPaddedAndFilledImageView.image.width + ' and height: ' + newPaddedAndFilledImageView.image.height);

    let newWavelet = new Wavelet(waveletType);

    let waveletKernel = newWavelet.getKernel(filterType);

    Console.warningln('waveletKernel: ' + waveletKernel);

    /*
    let array2D_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    let conv_matrix = [0, -1, 0,
        -1, 4, -1,
        0, -1, 0];

    let manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -4830, 8365, -2241, -1294, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
        
    manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
        
    manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -3.732050808, 6.464101615, -1.732050808, -1, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
    manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -3.732, 6.464, -1.732, -1, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
    manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -3.73, 6.46, -1.73, -1, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
    manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -0.482, 0.836, -0.224, -0.130, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
        
    manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -0.48, 0.84, -0.23, -0.13, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
        
    manualHi = [0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        -2, 3, -0.5, -0.5, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0];
        */
    // Console.warningln('linearKernel: ' + waveletKernel);

    // Console.warningln('manualHi: ' + manualHi);

    // let kernelSum = manualHi.reduce((a, b) => Math.round(a + b), 0);

    // let kernelSum = manualHi.reduce((accumulator, currentValue) => accumulator + currentValue, 0);

    // Console.warningln('kernelSum: ' + kernelSum);

    // let kernelMean = Math.round(kernelSum / manualHi.length);

    // Console.warningln('kernelMean: ' + kernelMean);

    // manualHi = manualHi.map(v => Math.round(v - kernelMean));

    /*
    Console.warningln('manualHi: ' + manualHi);

    var PConv = new Convolution;
    PConv.mode = Convolution.prototype.Image;
    PConv.sigma = 2.00;
    PConv.shape = 2.00;
    PConv.aspectRatio = 1.00;
    PConv.rotationAngle = 0.00;
    PConv.filterSource = "";
    PConv.rescaleHighPass = false;

    let highPassMode = 2;
    */

    let convolutionResult;

    if (axis === HORIZONTAL_AXIS) {
        convolutionResult = convolve2D(newPaddedAndFilledImageView, waveletKernel, axis, scaleConvolution);

        // convolutionResult = convolve2DReverseScaling(newPaddedAndFilledImageView, waveletKernel, axis, scaleConvolution);

        // newPaddedAndFilledImageView.image.convolve(manualHi, highPassMode = 2);
        // newPaddedAndFilledImageView.image.convolve(manualHi, 2);

        // Console.warningln('value at 8,8 of original: ' + view.image.sample(8, 8) + ' and 10, 10 of filtered: ' + newPaddedAndFilledImageView.image.sample(10, 10));
        /*
        let filterChannels = 1;
        let filterIsReal = true;
        let filterIsColor = false;

        let horizontalKernelWindow = new ImageWindow(newWavelet.filter_size, newWavelet.filter_size,
            filterChannels, // 1 channel for grayscale
            view.image.bitsPerSample,
            filterIsReal,
            filterIsColor,
            'horizontalKernelWindow'
        );

        // let kernelImage = Image(newWavelet.dec_hi);

        let horizontalKernelView = horizontalKernelWindow.mainView;
        horizontalKernelView.beginProcess(UndoFlag_NoSwapFile);
        let horizontalKernelImage = horizontalKernelView.image;
        horizontalKernelImage.apply(0);

        horizontalKernelWindow.show();
        
        for (let i = 0; i < newWavelet.filter_size; i++) {
            for (let j = 0; j < newWavelet.filter_size; j++) {
                horizontalKernelImage.setSample(waveletKernel[i][j], i, j);
            }
        }
        
        PConv.viewId = horizontalKernelView.id;

        PConv.executeOn(newPaddedAndFilledImageView);

        Console.warningln('value at 8,8 of original: ' + view.image.sample(8, 8) + ' and 10, 10 of filtered: ' + newPaddedAndFilledImageView.image.sample(10, 10));
        */
    }
    else {
        convolutionResult = convolve2D(newPaddedAndFilledImageView, waveletKernel, axis, scaleConvolution);

        // convolutionResult = convolve2DReverseScaling(newPaddedAndFilledImageView, waveletKernel, axis, scaleConvolution);
    }

    // return;

    newPaddedAndFilledImageView.window.forceClose();

    // newPaddedAndFilledImageView.image.convolve(newWavelet.dec_hi);

    // return;

    return convolutionResult;
}


function createStitchedAndNormalizedLevelImage(AA, AD, DA, DD, level) {
    // Create double sized stitched window
    let imageLabel = 'Level' + level;

    let stitchedImageWindow = new ImageWindow(AA.image.width * 2, AA.image.height * 2,
        AA.image.numberOfChannels, // 1 channel for grayscale
        AA.image.bitsPerSample,
        AA.image.isReal,
        AA.image.isColor,
        imageLabel
    );

    let stitchedImageView = stitchedImageWindow.mainView;
    stitchedImageView.beginProcess(UndoFlag_NoSwapFile);
    let stitchedImage = stitchedImageView.image;
    stitchedImage.apply(0);

    stitchedImageWindow.show();

    // Add AA with no scaling
    stitchedImageView.image.apply(AA.image, ImageOp_Screen, new Point(0, 0));

    // Add AD with scaling
    AD.image.rescale();
    stitchedImageView.image.apply(AD.image, ImageOp_Screen, new Point(AA.image.width, 0));

    // Add DA with scaling
    DA.image.rescale();
    stitchedImageView.image.apply(DA.image, ImageOp_Screen, new Point(0, AA.image.height));

    // Add DD with scaling
    DD.image.rescale();
    stitchedImageView.image.apply(DD.image, ImageOp_Screen, new Point(AA.image.width, AA.image.height));



    return stitchedImageView;
}



function wavedec2(view, paddingFillType, waveletType, levels) {
    let paddedImageWindow = new ImageWindow(view.image.width, view.image.height,
        view.image.numberOfChannels, // 1 channel for grayscale
        view.image.bitsPerSample,
        view.image.isReal,
        view.image.isColor
    );

    let paddedImageView = paddedImageWindow.mainView;
    paddedImageView.beginProcess(UndoFlag_NoSwapFile);
    let paddedImage = paddedImageView.image;
    paddedImage.apply(view.image);

    paddedImageWindow.show();

    currentImageView = paddedImageView;

    let newWavelet = new Wavelet(waveletType);

    let halfFilterSize = Math.floor(newWavelet.filter_size / 2);
    let padAmount = (halfFilterSize - 1) * 2;
    // padAmount = (halfFilterSize - 1);
    /*
    if (halfFilterSize % 2) {
        padAmount = padAmount + 1;
    }
    */

    Console.warningln('padAmount: ' + padAmount);

    // return;

    let levelsCoeffs = [];
    // newWavelet.printKernels();

    // return;

    for (let level = 0; level < levels; level++) {
        // Store the original width and height of image
        let originalWidth = currentImageView.image.width;
        let originalHeight = currentImageView.image.height;

        // Apply low - pass filtering in both horizontal and vertical directions to get the approximation image


        // A
        let axis = HORIZONTAL_AXIS;
        let filterType = 'dec_lo';
        let scaleConvolution = true;   // set true for any low pass filter that sums to nonzero

        let A = dwt(currentImageView, padAmount, 0, paddingFillType, waveletType, filterType, axis, scaleConvolution);

        // return;

        // D
        axis = HORIZONTAL_AXIS;
        filterType = 'dec_hi';
        scaleConvolution = false;

        let D = dwt(currentImageView, padAmount, 0, paddingFillType, waveletType, filterType, axis, scaleConvolution);

        // return;

        // AA
        axis = VERTICAL_AXIS;
        filterType = 'dec_lo';
        scaleConvolution = true;

        let AA = dwt(A, 0, padAmount, paddingFillType, waveletType, filterType, axis, scaleConvolution);

        // AD
        axis = VERTICAL_AXIS;
        filterType = 'dec_hi';
        scaleConvolution = false;

        let AD = dwt(A, 0, padAmount, paddingFillType, waveletType, filterType, axis, scaleConvolution);

        // DA
        axis = VERTICAL_AXIS;
        filterType = 'dec_lo';
        scaleConvolution = true;

        let DA = dwt(D, 0, padAmount, paddingFillType, waveletType, filterType, axis, scaleConvolution);

        // DD
        axis = VERTICAL_AXIS;
        filterType = 'dec_hi';
        scaleConvolution = false;

        let DD = dwt(D, 0, padAmount, paddingFillType, waveletType, filterType, axis, scaleConvolution);
        
        // let stitchedAndNormalizedLevelImage = createStitchedAndNormalizedLevelImage(AA, AD, DA, DD, level);

        
        // AD.window.forceClose();
        // DA.window.forceClose();
        // DD.window.forceClose();
        D.window.forceClose();
        A.window.forceClose();
        
        // horizontalHighPassView.window.forceClose();




        /*

        coeffs = [('', data)]
        for axis, wav, mode in zip(axes, wavelets, modes):
            new_coeffs = []
            for subband, x in coeffs:
                cA, cD = dwt_axis(x, wav, mode, axis)
                new_coeffs.extend([(subband + 'a', cA), (subband + 'd', cD)])
            coeffs = new_coeffs


            */

        /*
        #(See Note 1 below for details on filtering)
        [low_pass_horz, high_pass_horz] = filter_horizontal(current_level_image, wavelet_type)
        [low_pass_vert, high_pass_vert] = filter_vertical(low_pass_horz, wavelet_type)

        # 4. Subsample(downsample) the low - pass image
        approximation = subsample(low_pass_vert)  # This is the approximation at this level

        # 5. Calculate detail coefficients
        detail_horizontal = subband(current_level_image, high_pass_horz, 0)  # 1st subband(horizontal detail)
        detail_vertical = subband(current_level_image, high_pass_vert, 1)    # 2nd subband(vertical detail)
        detail_diagonal = subband(current_level_image, high_pass_vert, 2)  # 3rd subband(diagonal detail)

        # 6. Store the coefficients
        decomposition.append([approximation, detail_horizontal, detail_vertical, detail_diagonal])

        # 7. Prepare for the next level
        current_level_image = approximation  # Use the approximation from this level as input for the next level

        let padAmountHorizontal = 2;
        let padAmountVertical = 2;

        // let paddingFillType = PADDING_FILL_TYPE.PADDING_SYMMETRIC;
        let paddingFillType = 0;

        let waveletType = 'db2';

        let axis = 0;

        let newPaddedImageView = dwt(view, padAmountHorizontal, padAmountVertical, paddingFillType, waveletType, axis);
        */

        coeffs = {
            'AA': AA,
            'AD': AD,
            'DA': DA,
            'DD': DD,
            'originalWidth': originalWidth,
            'originalHeight': originalHeight
        };

        levelsCoeffs.push(coeffs);

        currentImageView = AA;
    }

    paddedImageWindow.forceClose();

    return levelsCoeffs;
}


function idwt(AA, AD, DA, DD, paddingFillType, waveletType, originalWidth, originalHeight) {
    let axis = HORIZONTAL_AXIS;

    let newWavelet = new Wavelet(waveletType);
    let lowPassRecKernel = newWavelet.getKernel('rec_lo');
    let highPassRecKernel = newWavelet.getKernel('rec_hi');

    // Console.warningln('lowPassRecKernel: ' + lowPassRecKernel);
    // Console.warningln('highPassRecKernel: ' + highPassRecKernel);

    // newWavelet.printKernels();

    // return;

    paddingFillType = PADDING_ZERO;

    let viewA = upsampleAndConvolveImage2X(AA, AD, axis, lowPassRecKernel, highPassRecKernel, paddingFillType);

    // return viewA;

    let viewD = upsampleAndConvolveImage2X(DA, DD, axis, lowPassRecKernel, highPassRecKernel, paddingFillType);

    axis = VERTICAL_AXIS;

    let reconstructedLevelView = upsampleAndConvolveImage2X(viewA, viewD, axis, lowPassRecKernel, highPassRecKernel, paddingFillType);

    viewA.window.forceClose();
    viewD.window.forceClose();

    let leftTopCrop = Math.round(lowPassRecKernel.length / 2) - 1;

    Console.warningln('leftTopCrop: ' + leftTopCrop);

    let widthCrop = -1 * (reconstructedLevelView.image.width - originalWidth);
    let heightCrop = -1 * (reconstructedLevelView.image.height - originalHeight);

    var PCrop = new Crop;

    PCrop.mode = Crop.prototype.AbsolutePixels;
    PCrop.xResolution = 72.000;
    PCrop.yResolution = 72.000;
    PCrop.metric = false;
    PCrop.forceResolution = false;
    PCrop.red = 0.000000;
    PCrop.green = 0.000000;
    PCrop.blue = 0.000000;
    PCrop.alpha = 1.000000;
    PCrop.noGUIMessages = false;
    PCrop.leftMargin = -1 * leftTopCrop;
    PCrop.topMargin = -1 * leftTopCrop;
    PCrop.rightMargin = widthCrop + leftTopCrop;
    PCrop.bottomMargin = heightCrop + leftTopCrop;

    PCrop.executeOn(reconstructedLevelView);

    return reconstructedLevelView;
}

function waverec2(levelsCoeffs, paddingFillType, waveletType) {
    let levels = levelsCoeffs.length;

    let currentImageView = levelsCoeffs[levels - 1].AA;

    for (let i = levels - 1; i >= 0; i--) {
        Console.warningln('Reconstructing level: ' + i);

        currentImageView = idwt(currentImageView, levelsCoeffs[i].DA, levelsCoeffs[i].AD, levelsCoeffs[i].DD, paddingFillType, waveletType, levelsCoeffs[i].originalWidth, levelsCoeffs[i].originalHeight);

        levelsCoeffs[i].AD.window.forceClose();
        levelsCoeffs[i].DA.window.forceClose();
        levelsCoeffs[i].DD.window.forceClose();

        // return currentImageView;

        // console.log(levelsCoeffs[i]);
    }

    return currentImageView;
}

