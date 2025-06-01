// #include "fft/fft.js"

function getShape(array) {
    if (!Array.isArray(array)) {
        return [];
    }
    const shape = [array.length];
    let current = array;
    while (Array.isArray(current[0])) {
        shape.push(current[0].length);
        current = current[0];
    }
    return shape;
}

function broadcastTo(array, shape) {
    if (!Array.isArray(array)) {
        throw new Error("Input must be an array.");
    }

    const originalShape = getShape(array);

    if (originalShape.length > shape.length) {
        throw new Error("Original array has more dimensions than target shape.");
    }

    for (let i = 0; i < originalShape.length; i++) {
        if (originalShape[originalShape.length - 1 - i] !== 1 && originalShape[originalShape.length - 1 - i] !== shape[shape.length - 1 - i]) {
            throw new Error("Array dimensions are incompatible for broadcasting.");
        }
    }

    const result = [];

    function recursiveBroadcast(currentArray, currentShapeIndex, currentResult) {
        if (currentShapeIndex === shape.length) {
            currentResult.push(array);
            return;
        }

        const currentShapeSize = shape[currentShapeIndex];

        if (currentShapeIndex >= shape.length - originalShape.length) {
            const originalSize = originalShape[originalShape.length - (shape.length - currentShapeIndex)]
            if (originalSize === 1) {
                for (let i = 0; i < currentShapeSize; i++) {
                    recursiveBroadcast(currentArray, currentShapeIndex + 1, currentResult);
                }
            } else {
                for (let i = 0; i < currentShapeSize; i++) {
                    recursiveBroadcast(currentArray[i], currentShapeIndex + 1, currentResult);
                }
            }
        } else {
            for (let i = 0; i < currentShapeSize; i++) {
                recursiveBroadcast(currentArray, currentShapeIndex + 1, currentResult);
            }
        }
    }

    recursiveBroadcast(array, 0, result);
    return result;
}

function notch(n, sigma) {
    /*
    Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
        n : int
        length of the gaussian notch filter
    sigma: float
        notch width

    Returns
    -------
        g : ndarray
            (n,) array containing the gaussian notch filter

    */

    if (n <= 0) {
        Console.warningln('n must be positive');

        return -1;
    }

    if (sigma <= 0) {
        Console.warningln('Sigma must be positive');

        return -1;
    }

    let g_list = [];

    for (let x = 0; x < n; x++) {
        let g = 1 - Math.exp(-1 * Math.pow(x, 2) / (2 * Math.pow(sigma, 2)));

        g_list.push(g);
    }

    return g_list;
}


function gaussian_filter(view, shape, sigma, rotateToVertical = false) {
    /*
    Create a gaussian notch filter

    Parameters
    ----------
        shape : tuple
        shape of the output filter
    sigma: float
        filter bandwidth

    Returns
    -------
        g : ndarray
        the impulse response of the gaussian notch filter

    */

    let g = notch(shape[shape.length - 1], sigma);
    let g_mask = broadcastTo(g, shape);

    let GaussianImageWindow = new ImageWindow(shape[shape.length - 1], shape[0],
        view.image.numberOfChannels, // 1 channel for grayscale
        view.image.bitsPerSample,
        view.image.isReal,
        view.image.isColor
    );

    let GaussianImageView = GaussianImageWindow.mainView;
    GaussianImageView.beginProcess(UndoFlag_NoSwapFile);
    let GaussianImage = GaussianImageView.image;
    GaussianImage.apply(0);

    for (let i = 0; i < GaussianImage.height; i++) {
        for (let j = 0; j < GaussianImage.width; j++) {
            if (rotateToVertical) {
                GaussianImage.setSample(g_mask[i][j], i, j);
            }
            else {
                GaussianImage.setSample(g_mask[i][j], j, i);
            }
        }
    }
    /*
    if (rotateToVertical) {
        GaussianImageView.image.rotate90cw();
    }
    */
    GaussianImageWindow.show();

    return GaussianImageView;
}

function rfft(input) {
    const n = input.length;
    const complexOutputLength = Math.floor(n / 2) + 1;
    const output = new Array(complexOutputLength).fill(null).map(() => ({ real: 0, imag: 0 }));

    // Basic DFT implementation (can be optimized)
    for (let k = 0; k < complexOutputLength; k++) {
        for (let i = 0; i < n; i++) {
            const angle = -2 * Math.PI * k * i / n;
            output[k].real += input[i] * Math.cos(angle);
            output[k].imag += input[i] * Math.sin(angle);
        }
    }
    return output
}

function createTestImage(view, shape, lineLocation) {
    let testImageWindow = new ImageWindow(shape[shape.length - 1], shape[0],
        view.image.numberOfChannels, // 1 channel for grayscale
        view.image.bitsPerSample,
        view.image.isReal,
        view.image.isColor
    );

    let testImageView = testImageWindow.mainView;
    testImageView.beginProcess(UndoFlag_NoSwapFile);
    let testImage = testImageView.image;
    testImage.apply(0);

    drawLine(testImageView, Math.round(testImage.width / 8), 0, Math.round(5 * testImage.width / 8), testImage.height, 0.8);
    
    for (let i = 0; i < testImage.height; i++) {
        testImage.setSample(1., lineLocation, i);
        testImage.setSample(1., i, lineLocation);
    }
    
    testImageWindow.show();

    return testImageView;
}

function drawLine(testImageView, x1, y1, x2, y2, value) {
    testImage = testImageView.image;

    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    const sx = x1 < x2 ? 1 : -1;
    const sy = y1 < y2 ? 1 : -1;
    let err = dx - dy;

    while (true) {
        if (x1 >= 0 && x1 < testImage.width && y1 >= 0 && y1 < testImage.height) {
            testImage.setSample(value, x1, y1);
        }
        if (x1 === x2 && y1 === y2) break;
        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

function fft2d(testImageView) {
    testImage = testImageView.image;

    const width = testImage.width;
    const height = testImage.height;

    // const rows = matrix.length;
    // const cols = matrix[0].length;

    // 1D FFT on rows
    const rowFFTs = matrix.map(row => fft(row));

    // Transpose
    const transposed = Array(cols).fill(null).map((_, i) =>
        rowFFTs.map(row => row[i])
    );

    // 1D FFT on transposed rows
    const colFFTs = transposed.map(row => fft(row));

    // Transpose back
    const result = Array(rows).fill(null).map((_, i) =>
        colFFTs.map(col => col[i])
    );

    return result;
}

function bitReverse(index, n) {
    let reversed = 0;
    for (let i = 0; i < n; i++) {
        if ((index >> i) & 1) {
            reversed |= (1 << (n - 1 - i));
        }
    }
    return reversed;
}

function fft1D(x) {
    const N = x.length;
    if (N <= 1) {
        return x;
    }

    const reversed = new Array(N);
    for (let i = 0; i < N; i++) {
        reversed[bitReverse(i, Math.log2(N))] = x[i];
    }

    const X = new Array(N);
    for (let i = 0; i < N; i++) {
        X[i] = { real: reversed[i].real, imag: reversed[i].imag };
    }

    for (let len = 2; len <= N; len *= 2) {
        const halfLen = len / 2;
        for (let i = 0; i < N; i += len) {
            for (let k = 0; k < halfLen; k++) {
                const w = {
                    real: Math.cos(-2 * Math.PI * k / len),
                    imag: Math.sin(-2 * Math.PI * k / len)
                };

                const even = X[i + k];
                const odd = {
                    real: X[i + k + halfLen].real * w.real - X[i + k + halfLen].imag * w.imag,
                    imag: X[i + k + halfLen].real * w.imag + X[i + k + halfLen].imag * w.real
                };

                X[i + k] = { real: even.real + odd.real, imag: even.imag + odd.imag };
                X[i + k + halfLen] = { real: even.real - odd.real, imag: even.imag - odd.imag };
            }
        }
    }

    return X;
}


function fft2D(data) {
    const rows = data.length;
    const cols = data[0].length;

    // 1. FFT on Rows
    let rowTransformed = [];
    for (let i = 0; i < rows; i++) {
        rowTransformed.push(fft1D(data[i]));
    }

    // 2. FFT on Columns
    let colTransformed = [];
    for (let j = 0; j < cols; j++) {
        let column = [];
        for (let i = 0; i < rows; i++) {
            column.push(rowTransformed[i][j]);
        }
        colTransformed.push(fft1D(column));
    }

    // Transpose the result to have the correct 2D structure
    let result = [];
    for (let i = 0; i < rows; i++) {
        let row = [];
        for (let j = 0; j < cols; j++) {
            row.push(colTransformed[j][i]);
        }
        result.push(row);
    }

    return result;
}

function fft2DPixinsight(fftImageView) {
    var FFT = new FourierTransform;
    FFT.radialCoordinates = false;
    FFT.centered = false;

    FFT.executeOn(fftImageView);

    return;
}

function ifft2DPixinsight() {
    var iFFT = new InverseFourierTransform;
    iFFT.idOfFirstComponent = "DFT_real";
    iFFT.idOfSecondComponent = "DFT_imaginary";
    iFFT.onOutOfRangeResult = InverseFourierTransform.prototype.DontCare;

    iFFT.executeGlobal();

    return;
}

function filterImage(imageView, filterView) {
    imageView.beginProcess(UndoFlag_NoSwapFile);

    for (let i = 0; i < imageView.image.height; i++) {
        for (let j = 0; j < imageView.image.width; j++) {
            imageView.image.setSample(imageView.image.sample(j, i) * filterView.image.sample(j, i), j, i);
        }
    }

    return;
}
