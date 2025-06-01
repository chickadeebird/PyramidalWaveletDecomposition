#feature-id PWDStarRepair : ChickadeeScripts > PyramidalWaveletDecomposition
#feature-icon  Pyramidal.svg
#feature-info This script takes a selected image and applies pyramidal wavelet decomposition.



#include <pjsr/StdButton.jsh>
#include <pjsr/StdIcon.jsh>
#include <pjsr/StdCursor.jsh>
#include <pjsr/Sizer.jsh>
#include <pjsr/FrameStyle.jsh>
#include <pjsr/NumericControl.jsh>
#include <pjsr/FileMode.jsh>
#include <pjsr/DataType.jsh>
#include <pjsr/ImageOp.jsh>
#include <pjsr/SampleType.jsh>
#include <pjsr/UndoFlag.jsh>
#include <pjsr/TextAlign.jsh>
#include <pjsr/FontFamily.jsh>
#include <pjsr/ColorSpace.jsh>
#include <pjsr/StarDetector.jsh>

#include "WaveletsCommon.js"
#include "PWDFiltering.js"

#define VERSION "v1.0.0 beta"
#define SCRIPTNAME "Pyramidal Wavelet Decomposition"



// Define platform-agnostic folder paths
let pathSeparator = (CoreApplication.platform == "MSWINDOWS" || CoreApplication.platform == "Windows") ? "\\" : "/";
let scriptTempDir = File.systemTempDirectory + pathSeparator + "PWDConfig";
let PWDConfigFile = scriptTempDir + pathSeparator + "PWD_config.csv";

// Ensure the temp directory exists
if (!File.directoryExists(scriptTempDir)) {
    File.createDirectory(scriptTempDir);
}

// Define global parameters
var PWDParameters = {
    targetView: undefined,
    targetWindow: undefined,
    totalNumberOfLevels: 8,
    levels: 1,
    waveletType: 'db2',
    PWDPythonToolsParentFolderPath: "",
    linearImage: true,

    save: function () {
        Parameters.set("linearImage", this.linearImage);
        Parameters.set("PWDPythonToolsParentFolderPath", this.PWDPythonToolsParentFolderPath);
        this.savePathToFile();
    },

    load: function () {
        if (Parameters.has("linearImage"))
            this.linearImage = Number(Parameters.getBoolean("linearImage"));
        if (Parameters.has("PWDPythonToolsParentFolderPath"))
            this.PWDPythonToolsParentFolderPath = Parameters.getString("PWDPythonToolsParentFolderPath");
        this.loadPathFromFile();
    },
    savePathToFile: function () {
        try {
            let file = new File;
            // console.writeln("Writing config file: " + PWDConfigFile);
            file.createForWriting(PWDConfigFile);
            file.outTextLn(this.PWDPythonToolsParentFolderPath);
            file.outTextLn(String(Number(this.linearImage)));
            file.close();
        } catch (error) {
            console.warningln("Failed to save PWDPythonTools parent folder path: " + error.message);
        }
    },

    loadPathFromFile: function () {
        try {
            if (File.exists(PWDConfigFile)) {
                let file = new File;
                // console.writeln("Reading config file: " + PWDConfigFile);
                file.openForReading(PWDConfigFile);
                let lines = File.readLines(PWDConfigFile);
                if (lines.length > 0) {
                    this.PWDPythonToolsParentFolderPath = lines[0].trim();
                    this.linearImage = Boolean(Number(lines[2].trim()));
                }
                file.close();
            }
        } catch (error) {
            console.warningln("Failed to load PWDPythonTools parent folder path: " + error.message);
        }
    }
};

function SpacedVerticalSizer() {
    this.__base__ = VerticalSizer;
    this.__base__();
    this.scaledSpacing = 5;
    this.scaledMargin = 5;
}

SpacedVerticalSizer.addItem = function (item, stretchFactor = 0) {
    //console.warningln("Adding " + item.constructor.name + " to " + this.constructor.name);
    if (item.constructor.name == "Sizer" && this.parentControl != null && this.parentControl.constructor.name != "Dialog") {
        item.clearMargin([]);
    }
    this.items.push(item);
    if (stretchFactor > 0) {
        this.add(item, stretchFactor);
    } else {
        this.add(item);
    }
}


// Dialog setup, image selection, etc.
function PWDDialog() {
    this.__base__ = Dialog;
    this.__base__();

    console.hide();
    PWDParameters.load();

    
    
    this.title = new Label(this);
    this.title.text = SCRIPTNAME + " - " + VERSION;
    this.title.textAlignment = TextAlign_Center;

    this.description = new TextBox(this);
    this.description.readOnly = true;
    this.description.height = 50;
    this.description.maxHeight = 50;
    this.description.text = "This script implements algorithms to detect and restore images containing artifacts caused by stars outside the field of view." +
        "";
    this.description.setMinWidth(640);
    // this.description.setMinHeight(150);

    this.imageSelectionLabel = new Label(this);
    this.imageSelectionLabel.text = "Select Image:";
    this.imageSelectionLabel.textAlignment = TextAlign_Right | TextAlign_VertCenter;

    this.imageSelectionDropdown = new ComboBox(this);
    this.imageSelectionDropdown.editEnabled = false;

    this.imageSelectionDropdown.onItemSelected = (index) => {
        if (index >= 0) {
            let window = ImageWindow.windowById(this.imageSelectionDropdown.itemText(index));
            if (window && !window.isNull) {
                PWDParameters.targetWindow = window;

                
            } 
        }
    };



    let windows = ImageWindow.windows;
    let activeWindowId = ImageWindow.activeWindow.mainView.id;
    PWDParameters.targetWindow = false;
    // Console.warningln("activeWindowId: " + activeWindowId);
    let previewCounter = 0;
    for (let i = 0; i < windows.length; ++i) {
        this.imageSelectionDropdown.addItem(windows[i].mainView.id);
        if (windows[i].mainView.id === activeWindowId) {
            this.imageSelectionDropdown.currentItem = i;
            PWDParameters.targetWindow = windows[i];
        }
    }

    

    // Console.warningln("targetWindow set to: " + PWDParameters.targetWindow);

    this.imageSelectionSizer = new HorizontalSizer;
    this.imageSelectionSizer.spacing = 4;
    this.imageSelectionSizer.add(this.imageSelectionLabel);
    this.imageSelectionSizer.add(this.imageSelectionDropdown, 100);


    this.levelsSelectionLabel = new Label(this);
    this.levelsSelectionLabel.text = "Select number of levels to decimate:";
    this.levelsSelectionLabel.textAlignment = TextAlign_Right | TextAlign_VertCenter;

    this.levelsSelectionDropdown = new ComboBox(this);
    this.levelsSelectionDropdown.editEnabled = false;

    this.levelsSelectionDropdown.onItemSelected = (index) => {
        PWDParameters.levels = this.levelsSelectionDropdown.itemText(index);
    };



    
    for (let i = 0; i < PWDParameters.totalNumberOfLevels; ++i) {
        let levelToAdd = i + 1;
        this.levelsSelectionDropdown.addItem(String(levelToAdd));
    }

    this.levelsSelectionSizer = new HorizontalSizer;
    this.levelsSelectionSizer.spacing = 4;
    this.levelsSelectionSizer.add(this.levelsSelectionLabel);
    this.levelsSelectionSizer.add(this.levelsSelectionDropdown, 100);


    // Wavelet selector

    this.waveletSelectionLabel = new Label(this);
    this.waveletSelectionLabel.text = "Select type of wavelet with which to decimate:";
    this.waveletSelectionLabel.textAlignment = TextAlign_Right | TextAlign_VertCenter;

    this.waveletSelectionDropdown = new ComboBox(this);
    this.waveletSelectionDropdown.editEnabled = false;

    this.waveletSelectionDropdown.onItemSelected = (index) => {
        PWDParameters.waveletType = this.waveletSelectionDropdown.itemText(index);
    };

    let newWavelet = new Wavelet('db2');
    let availableWaveletsList = newWavelet.getAvailableWaveletsList();
    
    for (let i = 0; i < availableWaveletsList.length; ++i) {
        this.waveletSelectionDropdown.addItem(availableWaveletsList[i]);

        // Set initial wavelet type to db2 for no particular reason
        if (availableWaveletsList[i] === PWDParameters.waveletType) {
            this.waveletSelectionDropdown.currentItem = i;
        }
    }

    this.waveletSelectionSizer = new HorizontalSizer;
    this.waveletSelectionSizer.spacing = 4;
    this.waveletSelectionSizer.add(this.waveletSelectionLabel);
    this.waveletSelectionSizer.add(this.waveletSelectionDropdown, 100);



    var dlg = this;

    
    

    this.linearCheckBox = new CheckBox(this);
    with (this.linearCheckBox) {
        toolTip = "Check if this is a linear image for star replacement with StarNet2"
        text = "Linear data";
        enabled = true;
        checked = PWDParameters.linearImage;
        bindings = function () {
            this.checked = PWDParameters.linearImage;
        }
        onCheck = function (value) {
            PWDParameters.linearImage = value;
        }
    }

    this.restoreButton = new PushButton(this);
    this.restoreButton.text = "Perform Wavelet Decimation and Restoration";
    this.restoreButton.onClick = function () {
        // var parent = this.parent;
        // var grandparent = parent.parent;
        // var greatgrandparent = grandparent.parent;
        // var denoiseAlgorithm = 1;
        Console.show();
        // let returnedDenoise = processImageAsLinear(PWDParameters.targetWindow.mainView, dlg, PWDParameters.linearImage);
        let returnedDenoise = processImage(PWDParameters.targetWindow.mainView, dlg);
        PWDParameters.save();
        Console.writeln('Image restored as linear');
        Console.hide();
    };

    this.restoreButtonSizer = new HorizontalSizer;
    this.restoreButtonSizer.spacing = 4;
    this.restoreButtonSizer.add(this.restoreButton);


    
        
    

    
    this.lblState = new Label(this);
    this.lblState.text = "";

    this.lblStateSizer = new HorizontalSizer;
    this.lblStateSizer.spacing = 4;
    this.lblStateSizer.add(this.lblState);

    var progressValue = 0;

    this.progressBar = new Label(this);
    with (this.progressBar) {
        lineWidth = 1;
        frameStyle = FrameStyle_Box;
        textAlignment = TextAlign_Center | TextAlign_VertCenter;

        onPaint = function (x0, y0, x1, y1) {
            var g = new Graphics(dlg.progressBar);
            g.fillRect(x0, y0, x1, y1, new Brush(0xFFFFFFFF));
            if (progressValue > 0) {
                var l = (x1 - x0 + 1) * progressValue;
                g.fillRect(x0, y0, l, y1, new Brush(0xFF00EFE0));
            }
            g.end();
            text = (progressValue * 100).toFixed(0) + "%";
        }
    }

    this.progress = function (n) {
        progressValue = n;// Math.min(n, 1);
        dlg.progressBar.repaint();
    }

    // Wrench Icon Button for setting the SetiAstroDenoise parent folder path
    this.setupButton = new ToolButton(this);
    this.setupButton.icon = this.scaledResource(":/icons/wrench.png");
    this.setupButton.setScaledFixedSize(24, 24);
    this.setupButton.onClick = function () {
        let pathDialog = new GetDirectoryDialog;
        // let pathDialog = new OpenFileDialog;
        pathDialog.initialPath = PWDParameters.PWDPythonToolsParentFolderPath;
        if (pathDialog.execute()) {
            PWDParameters.PWDPythonToolsParentFolderPath = pathDialog.directory;
            PWDParameters.save();
            

        }
    };
    
    // New Instance button
    this.newInstanceButton = new ToolButton(this);
    this.newInstanceButton.icon = this.scaledResource(":/process-interface/new-instance.png");
    this.newInstanceButton.setScaledFixedSize(24, 24);
    this.newInstanceButton.toolTip = "Save a new instance of this script";
    this.newInstanceButton.onMousePress = function () {
        this.dialog.newInstance();
    }.bind(this);

    this.undoRepairButton = new PushButton(this);
    this.undoRepairButton.text = "Undo";
    this.undoRepairButton.toolTip = "Undo the last repair";
    this.undoRepairButton.icon = ":/icons/undo.png";
    this.undoRepairButton.onClick = () => {
        if (PWDParameters.targetWindow && !PWDParameters.targetWindow.isNull) {
            PWDParameters.targetWindow.undo();
        } else {
            console.writeln("No valid window selected for undo!");
        }
    };

    this.redoRepairButton = new PushButton(this);
    this.redoRepairButton.text = "Redo";
    this.redoRepairButton.toolTip = "Redo the last repair";
    this.redoRepairButton.icon = ":/icons/redo.png";
    this.redoRepairButton.onClick = () => {
        if (PWDParameters.targetWindow && !PWDParameters.targetWindow.isNull) {
            PWDParameters.targetWindow.redo();
        } else {
            console.writeln("No valid window selected for redo!");
        }
    };

    this.buttonsSizer = new HorizontalSizer;
    this.buttonsSizer.spacing = 6;
    this.buttonsSizer.add(this.newInstanceButton);
    this.buttonsSizer.add(this.setupButton);
    this.buttonsSizer.addStretch();
    this.buttonsSizer.add(this.undoRepairButton);
    this.buttonsSizer.add(this.redoRepairButton);
    // this.buttonsSizer.addStretch();
    // this.buttonsSizer.add(this.okButton);
    // this.buttonsSizer.add(this.cancelButton);
    // this.buttonsSizer.addStretch();







    // Layout
    this.leftSizer = new VerticalSizer;
    this.leftSizer.margin = 6;
    this.leftSizer.spacing = 6;
    this.leftSizer.addStretch();
    this.leftSizer.add(this.title);
    this.leftSizer.add(this.description);
    this.leftSizer.addStretch();
    this.leftSizer.add(this.imageSelectionSizer);
    this.leftSizer.spacing = 6;
    this.leftSizer.add(this.levelsSelectionSizer);
    this.leftSizer.spacing = 6;
    this.leftSizer.add(this.waveletSelectionSizer);
    this.leftSizer.spacing = 6;
    this.leftSizer.add(this.linearCheckBox);
    this.leftSizer.spacing = 6;
    this.leftSizer.add(this.restoreButtonSizer);
    this.leftSizer.spacing = 6;
    this.leftSizer.add(this.lblStateSizer);
    this.leftSizer.spacing = 8;
    this.leftSizer.add(this.progressBar);
    this.leftSizer.spacing = 6;
    
    this.leftSizer.addStretch();


    this.sizer = new HorizontalSizer;
    this.sizer.spacing = 4;

    // Add the leftSizer and the previewSizer to the mainSizer
    this.sizer.add(this.leftSizer);



    // this.sizer.add(this.setupButton);
    this.leftSizer.addSpacing(12);
    this.leftSizer.add(this.buttonsSizer);

    this.windowTitle = SCRIPTNAME;
    this.adjustToContents();


}
PWDDialog.prototype = new Dialog;

function saveImageAsXISF(inputFolderPath, view) {
    // Obtain the ImageWindow object from the view's main window
    let imgWindow = view.isMainView ? view.window : view.mainView.window;

    if (!imgWindow) {
        throw new Error("Image window is undefined for the specified view.");
    }

    let fileName = imgWindow.mainView.id;  // Get the main view's id as the filename
    let filePath = inputFolderPath + pathSeparator + fileName + ".xisf";

    // Set the image format to 32-bit float if not already set
    imgWindow.bitsPerSample = 32;
    imgWindow.ieeefpSampleFormat = true;

    // Save the image in XISF format
    if (!imgWindow.saveAs(filePath, false, false, false, false)) {
        throw new Error("Failed to save image as 32-bit XISF: " + filePath);
    }

    console.writeln("Image saved as 32-bit XISF: " + filePath);
    return filePath;
}








// Main execution block for running the script
let dialog = new PWDDialog();
console.show();
Console.warningln("                  .-.");
Console.warningln("                 (  `>");
Console.warningln("                 /  \\");
Console.warningln("                /  \\ |");
Console.warningln("               / ,_//");
Console.warningln("      ~~~~~~~~//`--`~~~~~~~");
Console.warningln("      ~~~~~~~//~~~~~~~~~~~");
Console.warningln("            `");
Console.criticalln("      Chickadee Scripts - " + SCRIPTNAME);
console.flush();

if (dialog.execute()) {
    let selectedIndex = dialog.imageSelectionDropdown.currentItem;
    let selectedView = ImageWindow.windows[selectedIndex];

    if (!selectedView) {
        console.criticalln("Please select an image.");
    } else {
        let artifactCorrectionFileWindow = ImageWindow.open(PWDParameters.artifactCorrectionImageFile)[0];
        if (artifactCorrectionFileWindow) {
            artifactCorrectionFileWindow.show();
        }
    }
}



function processImage(view, dlg) {
    /*
    let testSignal = [1, 2, 3, 4, 5, 6, 7, 8];
    const signal = [
        { real: 1, imag: 0 },
        { real: 2, imag: 0 },
        { real: 1, imag: 0 },
        { real: 0, imag: 0 },
        { real: 1, imag: 0 },
        { real: 2, imag: 0 },
        { real: 1, imag: 0 },
        { real: 0, imag: 0 },
    ];

    let fftTestSignal = fft1D(signal);

    Console.warningln('testSignal: ' + testSignal);
    Console.warningln('fftTestSignal: ' + fftTestSignal);

    return;
    */
    let testImageShape = [256, 256];
    let lineLocation = 100;
    let testImageView = createTestImage(view, testImageShape, lineLocation);

    // return;

    let sigma = 8;
    let widthFraction = sigma / view.image.width;
    // let shape = [view.image.width, view.image.height];
    let s = view.image.width * widthFraction;
    let rotateToVertical = true;

    let GaussianFilterView = gaussian_filter(view, testImageShape, s, rotateToVertical);

    fft2DPixinsight(testImageView);

    let realFFTView = View.viewById('DFT_real');
    let imaginaryFFTView = View.viewById('DFT_imaginary');

    filterImage(realFFTView, GaussianFilterView);
    filterImage(imaginaryFFTView, GaussianFilterView);

    ifft2DPixinsight();

    return;

    // Console.warningln('Gaussian filter: ' + gFilter);

    var FFT = new FourierTransform;
    FFT.radialCoordinates = false;
    FFT.centered = false;

    FFT.executeOn(view);



    var iFFT = new InverseFourierTransform;
    iFFT.idOfFirstComponent = "DFT_real";
    iFFT.idOfSecondComponent = "DFT_imaginary";
    iFFT.onOutOfRangeResult = InverseFourierTransform.prototype.DontCare;

    iFFT.executeGlobal();

    return;

    let intermediateImageView = view;
    
    // dlg.lblState.text = 'Resampling completed - initiating StarNet2 operation to remove stars';
    // processEvents();

    // let paddingFillType = PADDING_FILL_TYPE.PADDING_SYMMETRIC;
    let paddingFillType = PADDING_SYMMETRIC;

    let waveletType = PWDParameters.waveletType;

    // Console.warningln('waveletType: ' + waveletType);

    let levels = PWDParameters.levels;

    // Console.warningln('waveletType: ' + waveletType + ' levels: ' + levels);

    dlg.lblState.text = 'Decimation using ' + waveletType + ' to ' + levels + ' levels';
    processEvents();

    let waveletDecompositionCoefficients = wavedec2(view, paddingFillType, waveletType, levels);

    // return;

    // let AA = waveletDecompositionCoefficients.AA;

    // Console.warningln('New AA decomposition width: ' + waveletDecompositionCoefficients.AA.image.width + ' height: ' + waveletDecompositionCoefficients.AA.image.height);
    // Console.warningln('New AD decomposition width: ' + waveletDecompositionCoefficients.AD.image.width + ' height: ' + waveletDecompositionCoefficients.AD.image.height);

    // return;

    let reconstructedLevelView = waverec2(waveletDecompositionCoefficients, paddingFillType, waveletType);

    // let reconstructedLevelView = idwt(waveletDecompositionCoefficients[0].AA, waveletDecompositionCoefficients[0].DA, waveletDecompositionCoefficients[0].AD, waveletDecompositionCoefficients[0].DD, paddingFillType, waveletType);

    // let newPaddedImageView = dwt(view, padAmountHorizontal, padAmountVertical, paddingFillType, waveletType, axis);
    
    return reconstructedLevelView;
}



function processImageAsLinear(view, dlg, processAsLinear) {
    
    // return;

    let intermediateImageWindow = new ImageWindow(view.image.width, view.image.height,
        view.image.numberOfChannels, // 1 channel for grayscale
        view.image.bitsPerSample,
        view.image.isReal,
        view.image.isColor
    );

    let intermediateImageView = intermediateImageWindow.mainView;
    intermediateImageView.beginProcess(UndoFlag_NoSwapFile);
    let intermediateImageImage = intermediateImageView.image;
    intermediateImageImage.apply(view.image);

    intermediateImageWindow.show();
    /*
    var PResample = new Resample;
    PResample.xSize = 0.4000;
    PResample.ySize = 0.4000;
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

    dlg.lblState.text = 'Resampling image to optimize efficiency';
    processEvents();

    PResample.executeOn(intermediateImageView);
    */

    if (processAsLinear) {
        dlg.lblState.text = 'Applying temporary auto STF';
        processEvents();

        let constantCalcWindow = new ImageWindow(view.image.width, view.image.height,
            view.image.numberOfChannels, // 1 channel for grayscale
            view.image.bitsPerSample,
            view.image.isReal,
            view.image.isColor
        );

        let constantCalcView = constantCalcWindow.mainView;
        constantCalcView.beginProcess(UndoFlag_NoSwapFile);
        let constantCalcImage = constantCalcView.image;
        constantCalcImage.apply(view.image);

        var P = new PixelMath;
        P.expression =
            "C = -2.8  ;\n" +
            "B = 0.20  ;\n" +
            "c = min(max(0,med($T)+C*1.4826*mdev($T)),1);\n" +
            "mtf(mtf(B,med($T)-c),max(0,($T-c)/~c))";
        P.expression1 = "";
        P.expression2 = "";
        P.expression3 = "";
        P.useSingleExpression = true;
        P.symbols = "C,B,c";
        P.clearImageCacheAndExit = false;
        P.cacheGeneratedImages = false;
        P.generateOutput = true;
        P.singleThreaded = false;
        P.optimization = true;
        P.use64BitWorkingImage = false;
        P.rescale = false;
        P.rescaleLower = 0;
        P.rescaleUpper = 1;
        P.truncate = true;
        P.truncateLower = 0;
        P.truncateUpper = 1;
        P.createNewImage = false;
        P.showNewImage = true;
        P.newImageId = "";
        P.newImageWidth = 0;
        P.newImageHeight = 0;
        P.newImageAlpha = false;
        P.newImageColorSpace = PixelMath.prototype.SameAsTarget;
        P.newImageSampleFormat = PixelMath.prototype.SameAsTarget;


        P.expression =
            "C = -2.8  ;\n" +
            "B = 0.25  ;\n" +
            "min(max(0,med($T)+C*1.4826*mdev($T)),1);";

        P.executeOn(constantCalcView);

        var C0 = constantCalcImage.sample(0, 0);

        Console.writeln("C0: " + C0);

        constantCalcImage.apply(view.image);

        var targetBackground = 0.25;

        P.expression = "mtf(" + targetBackground + ",med($T) - " + C0 + ");";

        P.executeOn(constantCalcView);

        var mtfConstant = constantCalcImage.sample(0, 0);

        Console.writeln("mtfConstant: " + mtfConstant);



        // ****************** Start here with the input image after the constants have been calculated

        intermediateImageImage.apply(view.image);

        P.expression = "max(0, ($T - " + C0 + ")/~" + C0 + ");";

        P.executeOn(intermediateImageView);

        P.expression = "mtf((" + mtfConstant + "), $T);";

        P.executeOn(intermediateImageView);

        // ****************** Do the operations on the autoSTF image here

        // -------->

        constantCalcWindow.show();
        constantCalcWindow.forceClose();
        // autoSTFWindow.forceClose();
    }

    dlg.lblState.text = 'Processing image';
    processEvents();

    //
    //
    //
    // ********************* Put processing code here
    // processImageWaveletFFT(intermediateImageView, dlg);
    processImage(intermediateImageView, dlg);

    intermediateImageView.window.forceClose();

    return;
    //
    //
    //

    if (processAsLinear) {
        dlg.lblState.text = 'Reversing auto STF';
        processEvents();

        // ****************** Now reverse out of the autoSTF

        P.expression = "mtf((1 - " + mtfConstant + "), $T);";

        P.executeOn(intermediateImageView);

        P.expression = "max(0, ($T * ~" + C0 + ") + " + C0 + ");";

        P.executeOn(intermediateImageView);
    }

    var P = new PixelMath;
    P.expression =
        "" + intermediateImageView.id + "";
    P.expression1 = "";
    P.expression2 = "";
    P.expression3 = "";
    P.useSingleExpression = true;
    P.symbols = "C,B,c";
    P.clearImageCacheAndExit = false;
    P.cacheGeneratedImages = false;
    P.generateOutput = true;
    P.singleThreaded = false;
    P.optimization = true;
    P.use64BitWorkingImage = false;
    P.rescale = false;
    P.rescaleLower = 0;
    P.rescaleUpper = 1;
    P.truncate = true;
    P.truncateLower = 0;
    P.truncateUpper = 1;
    P.createNewImage = false;
    P.showNewImage = true;
    P.newImageId = "";
    P.newImageWidth = 0;
    P.newImageHeight = 0;
    P.newImageAlpha = false;
    P.newImageColorSpace = PixelMath.prototype.SameAsTarget;
    P.newImageSampleFormat = PixelMath.prototype.SameAsTarget;
    /*
    if (PWDParameters.targetMaskWindow) {
        let maskView = View.viewById(dlg.maskSelectionDropdown.itemText(dlg.maskSelectionDropdown.currentItem));

        P.expression =
            "" + intermediateImageView.id + "*~" + maskView.id + "+" + view.id + "*" + maskView.id;

        P.executeOn(intermediateImageView);
    }
    */




    P.expression =
        "" + intermediateImageView.id + "";

    P.executeOn(view);

    intermediateImageWindow.forceClose();

    dlg.lblState.text = 'Image restored';

    return intermediateImageView;
}

