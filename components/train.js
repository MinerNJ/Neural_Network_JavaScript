//Bringing in BrainJS libraries and the MNIST dataset.
const fs = require('fs');
const mnist = require('./mnist');
const brain = require('brain.js');


async function run() {

    //Calling the images and labels from mnist.js.

    const trainLabels = await mnist.getTrainLabels();
    const trainDigits = await mnist.getTrainImages();

    const testLabels = await mnist.getTestLabels();
    const testDigits = await mnist.getTestImages();

    const labels = [...trainLabels, ...testLabels];
    const digits = [...trainDigits, ...testDigits];

    //Resizing images and lables to a binary array format that BrainJS understands.

    const imageSize = 28 * 28;
    const trainingData = [];

    for (let ix = 0; ix < labels.length; ix++) {
        const start = ix * imageSize;
        const input = digits.slice(start, start + imageSize).map(mnist.normalize);
        const output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        output[labels[ix]] = 1;
        trainingData.push({ input, output });
    };

    //Declaring hidden layer of 532 nodes ((2/3*trainingData) + 10). Input 
    //and output layers declared above.

    const netOptions = {
        hiddenLayers: [532]
    };

    //Declaring the number of iterations the model should perform, console logging
    //the error and bias weights for each iteration.

    const trainingOptions = {
        iterations: 20000,
    };

    //Creating the neural network and feeding the training data in. CrossValidate is
    //used since it creates multiple neural networks internally and selects the 
    //best one to be used for the model.

    const crossValidate = new brain.CrossValidate(brain.NeuralNetwork, netOptions);
    const stats = crossValidate.train(trainingData, trainingOptions);
    const net = crossValidate.toNeuralNetwork();

    //Once the best model is selected it is turned into a JSON object and written to 
    //the console for inspection. 

    const model = net.toJSON();
    fs.writeFile('./data/model.json', JSON.stringify(model), 'utf8', () => console.log(stats));
}

run();