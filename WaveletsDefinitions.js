#define KERNEL_DIRECTION_HORIZONTAL 0
#define KERNEL_DIRECTION_VERTICAL 1


function Wavelet(name) {
    this.__base__ = Object;
    this.__base__(name);

    this.dec_lo = [-999];

    this.name = name;

    function daubechiesHighPass(lowPassFilter) {
        const N = lowPassFilter.length;
        const highPassFilter = new Array(N);

        for (let n = 0; n < N; n++) {
            highPassFilter[n] = Math.pow(-1, n) * lowPassFilter[N - 1 - n];
        }

        return highPassFilter;
    }

    function daubechiesRecoveryLowPass(lowPassFilter) {
        // ***********

        const recoveryFilter = [...lowPassFilter].reverse();
        // const recoveryFilter = lowPassFilter;

        /*
        for (let i = 0; i < recoveryFilter.length; i += 2) {
            recoveryFilter[i] = -recoveryFilter[i];
        }
        */
        return recoveryFilter;
    }

    function calculateRecoveryHighPass(lowPassFilter) {
        const highPassFilter = daubechiesHighPass(lowPassFilter);
        const recHighPassFilter = [...highPassFilter].reverse();
        // const recHighPassFilter = highPassFilter;
        /*
        const N = h.length;
        const g_recovery = [];

        for (let n = 0; n < N; n++) {
            g_recovery[n] = Math.pow(-1, n) * h[N - 1 - n];
        }
        */
        return recHighPassFilter;
    }


    if (name === 'db2') {
        
        this.dec_lo = [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604];
        // this.dec_lo = [-0.1294095225512604, 0.2241438680420134, 0.8365163037378079, 0.4829629131445341];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

        /*
        this.dec_hi = [-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037];
        this.rec_lo = [0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037];
        this.rec_hi = [-0.12940952255126037, -0.2241438680420134, 0.8365163037378079, -0.48296291314453416];
        */
        /*
        this.dec_lo = [-0.12940952, 0.22414386, 0.83651630, 0.48296292];
        this.dec_hi = [-0.4830, 0.8365, -0.2241, -0.1294];
        this.rec_lo = [0.48296292, 0.83651630, 0.22414386, -0.12940952];
        this.rec_hi = [-0.12940952, -0.22414386, 0.83651630, -0.48296292];
        */

    }

    if (name === 'db3') {
        
        this.dec_lo = [0.3326705529500825,
            0.8068915093110924,
            0.4598775021184914,
            -0.1350110200102546,
            -0.0854412738820267,
            0.0352262918857095];
            
        // this.dec_lo = [0.03522629188570953, -0.08544127388202666, -0.13501102001025458, 0.45987750211849154, 0.8068915093110925, 0.33267055295008263];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'db4') {

        this.dec_lo = [0.230377813309,
            0.714846570553,
            0.630880766793,
            -0.027983769417,
            -0.187034811719,
            0.030841381836,
            0.032883011667,
            -0.010597401785];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'db5') {

        this.dec_lo = [0.1601023979741929,
            0.6038292697971895,
            0.7243085284377726,
            0.1384281459013203,
            -0.2422948870663823,
            -0.0322448695846381,
            0.0775714938400459,
            -0.0062414902127983,
            -0.0125807519990820,
            0.0033357252854738];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'db6') {

        this.dec_lo = [0.111540743350,
            0.494623890398,
            0.751133908021,
            0.315250351709,
            -0.226264693965,
            -0.129766867567,
            0.097501605587,
            0.027522865530,
            -0.031582039318,
            0.000553842201,
            0.004777257511,
            -0.001077301085];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'db7') {

        this.dec_lo = [0.077852054085,
            0.396539319482,
            0.729132090846,
            .469782287405,
            -0.143906003929,
            -0.224036184994,
            .071309219267,
            0.080612609151,
            -0.038029936935,
            -0.016574541631,
            0.012550998556,
            0.000429577973,
            -0.001801640704,
            0.000353713800];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'db8') {

        this.dec_lo = [0.054415842243,
            0.312871590914,
            0.675630736297,
            0.585354683654,
            -0.015829105256,
            -0.284015542962,
            0.000472484574,
            0.128747426620,
            -0.017369301002,
            -0.044088253931,
            0.013981027917,
            0.008746094047,
            -.0004870352993,
            -0.000391740373,
            0.000675449406,
            -0.000117476784];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'db9') {

        this.dec_lo = [0.038077947364,
            0.243834674613,
            0.604823123690,
            0.657288078051,
            0.133197385825,
            -0.293273783279,
            -0.096840783223,
            0.148540749338,
            0.030725681479,
            -0.067632829061,
            0.000250947115,
            0.022361662124,
            -0.004723204758,
            -0.004281503682,
            0.001847646883,
            0.000230385764,
            -0.000251963189,
            0.000039347320];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'db10') {

        this.dec_lo = [0.026670057901,
            0.188176800078,
            0.527201188932,
            0.688459039454,
            0.281172343661,
            -0.249846424327,
            -0.195946274377,
            0.127369340336,
            0.093057364604,
            -0.071394147166,
            -0.029457536822,
            0.033212674059,
            0.003606553567,
            -0.010733175483,
            0.001395351747,
            0.001992405295,
            -0.000685856695,
            -0.000116466855,
            0.000093588670,
            -0.000013264203];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'sym4') {

        this.dec_lo = [-0.107148901418,
            -0.041910965125,
            0.703739068656,
            1.136658243408,
            0.421234534204,
            -0.140317624179,
            -0.017824701442,
            0.045570345896];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'sym5') {

        this.dec_lo = [0.038654795955,
            0.041746864422,
            -0.055344186117,
            0.281990696854,
            1.023052966894,
            0.896581648380,
            0.023478923136,
            -0.247951362613,
            -0.029842499869,
            0.027632152958];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'sym6') {

        this.dec_lo = [0.021784700327,
            0.004936612372,
            -0.166863215412,
            -0.068323121587,
            0.694457972958,
            1.113892783926,
            0.477904371333,
            -0.102724969862,
            -0.029783751299,
            0.063250562660,
            0.002499922093,
            -0.011031867509];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'coif1') {

        this.dec_lo = [0.038580777748,
            -0.126969125396,
            -0.077161555496,
            0.607491641386,
            0.745687558934,
            0.226584265197];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    if (name === 'coif2') {

        this.dec_lo = [0.016387336463,
            -0.041464936782,
            -0.067372554722,
            0.386110066823,
            0.812723635450,
            0.417005184424,
            -0.076488599078,
            -0.059434418646,
            0.023680171947,
            0.005611434819,
            -0.001823208871,
            -0.000720549445];
        this.dec_hi = daubechiesHighPass(this.dec_lo);
        this.rec_lo = daubechiesRecoveryLowPass(this.dec_lo);
        this.rec_hi = calculateRecoveryHighPass(this.dec_lo);

    }

    this.availableWaveletsList = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'sym4', 'sym5', 'sym6', 'coif1', 'coif2'];

    this.filter_size = this.rec_hi.length;

    if (this.dec_lo[0] === -999) {
        Console.warningln('Wavelet name provided not in wavelet list: ' + name);
    }

    this.getAvailableWaveletsList = function () {
        return this.availableWaveletsList;
    };

    this.getKernel = function (kernelType) {
        let linearKernel = [];

        if (kernelType === 'dec_lo') {
            linearKernel = [...this.dec_lo];
        }
        else if (kernelType === 'dec_hi') {
            linearKernel = [...this.dec_hi];
        }
        else if (kernelType === 'rec_lo') {
            linearKernel = [...this.rec_lo];
        }
        else {
            linearKernel = [...this.rec_hi];
        }
        /*
        if (!(linearKernel % 2)) {
            linearKernel.push(0);
        }
        */
        return linearKernel;
    };

    this.getKernelArray = function (kernelType, direction) {
        let linearKernel = this.getKernel(kernelType);

        let kernelLength = linearKernel.length;

        let kernelPadding = Math.floor(kernelLength / 2);

        let kernelArray = [];

        for (let i = 0; i < kernelLength; i++) {
            kernelArray[i] = [];
            for (let j = 0; j < kernelLength; j++) {
                kernelArray[i][j] = 0; // Initialize with default value of 0
            }
        }

        // let kernelArray = Array(kernelLength).fill().map(() => Array(kernelLength).fill(0));

        if (direction === KERNEL_DIRECTION_VERTICAL) {
            for (let kernelElement = 0; kernelElement < kernelLength; kernelElement++) {
                kernelArray[kernelElement][kernelPadding] = linearKernel[kernelElement];
            }
        }
        else {
            for (let kernelElement = 0; kernelElement < kernelLength; kernelElement++) {
                kernelArray[kernelPadding][kernelElement] = linearKernel[kernelElement];
            }
        }

        // Now flatten the array
        const flattenedKernelArray = [];
        for (let i = 0; i < kernelLength; i++) {
            for (let j = 0; j < kernelLength; j++) {
                flattenedKernelArray.push(kernelArray[i][j]);
            }
        }

        return flattenedKernelArray;
    };

    this.printKernels = function () {
        Console.warningln('Wavelet name: ' + this.name);
        Console.warningln('Lo pass dec: ' + this.dec_lo);
        Console.warningln('Hi pass dec: ' + this.dec_hi);
        Console.warningln('Lo pass rec: ' + this.rec_lo);
        Console.warningln('Hi pass rec: ' + this.rec_hi);

        return;
    };

}


