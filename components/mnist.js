//Importing required libraries.
const fs = require('fs');
const http = require('http');
const zlib = require('zlib');

//Gathering data from MNIST webiste.
const dataDir = './data/';
const mnistBaseURL = 'http://yann.lecun.com/exdb/mnist/';
const trainImagesFile = 'train-images-idx3-ubyte';
const trainLabelsFile = 'train-labels-idx1-ubyte';
const testImagesFile = 't10k-images-idx3-ubyte';
const testLabelsFile = 't10k-labels-idx1-ubyte';

//Writing data into corresponding variables.

async function downloadMnistDataset(fileName) {
    if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir);
    }

    if (!fs.existsSync(dataDir + fileName)) {
        const gzFile = dataDir + fileName + '.gz';
        if (!fs.existsSync(gzFile)) {
            const outputStream = fs.createWriteStream(gzFile);
            const response = await httpGetPromisified(mnistBaseURL + fileName + '.gz');
            response.pipe(outputStream);
            await new Promise(resolve => outputStream.on('finish', () => outputStream.close(() => resolve())));
        }

        await gunzip(gzFile, dataDir + fileName);
    }
}

//Download all MNIST files specified

function downloadAll() {
    return Promise.all([trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile].map(downloadMnistDataset));
}

//Getting promise for data download

function httpGetPromisified(url) {
    return new Promise((resolve, reject) => http.get(url, res => resolve(res)));
}

//Compressing files for read/write.

function gunzip(compressedFile, uncompressedFile) {
    const compressed = fs.createReadStream(compressedFile);
    const uncompressed = fs.createWriteStream(uncompressedFile);

    return new Promise((resolve, reject) => {
        compressed.pipe(zlib.createGunzip()).pipe(uncompressed).on('finish', err => {
            if (err) {
                reject(err);
            } else {
                resolve();
            }
        });
    });
}

//Reading in categories for data.
async function readLabels(fileName) {
    const stream = fs.createReadStream(dataDir + fileName, { highWaterMark: 32 * 1024 });
    let firstChunk = true;

    const labels = [];

    for await (const chunk of stream) {
        let start = 0;
        if (firstChunk) {
            const version = chunk.readInt32BE(0);
            if (version !== 2049) {
                throw "label file: wrong format"
            }
            start = 8;
            firstChunk = false;
        }
        for (let i = start; i < chunk.length; i++) {
            labels.push(chunk.readUInt8(i));
        }
    }

    return labels;
}

//Reading in images from data.

async function readImages(fileName) {
    const stream = fs.createReadStream(dataDir + fileName, { highWaterMark: 32 * 1024 });
    let firstChunk = true;

    const digits = [];

    for await (const chunk of stream) {
        let start = 0;
        if (firstChunk) {
            const version = chunk.readInt32BE(0);
            if (version !== 2051) {
                throw "label file: wrong format"
            }
            start = 16;
            firstChunk = false;
        }
        for (let i = start; i < chunk.length; i++) {
            digits.push(chunk.readUInt8(i));
        }
    }

    return digits;
}

//Mixing up the data array to help with bias.

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

//Return shuffled indices, get all the testing and training images and
//labels, then normalize the data into binary by dividing the image size
//by the total number of pixels.

function getShuffledIndexes(noOfEntries) {
    const array = [];
    for (let i = 0; i < noOfEntries; i++) {
        array[i] = i;
    }
    shuffleArray(array);
    return array;
}

async function getTrainImages() {
    await downloadAll();
    return readImages(trainImagesFile);
}

async function getTrainLabels() {
    await downloadAll();
    return readLabels(trainLabelsFile);
}

async function getTestImages() {
    await downloadAll();
    return readImages(testImagesFile);
}

async function getTestLabels() {
    await downloadAll();
    return readLabels(testLabelsFile);
}

function normalize(num) {
    if (num != 0) {
        return num / 255;
    }
    return 0;
}

//Export for train.js and run.js

module.exports = {
    getTrainImages,
    getTrainLabels,
    getTestImages,
    getTestLabels,
    getShuffledIndexes,
    normalize
}