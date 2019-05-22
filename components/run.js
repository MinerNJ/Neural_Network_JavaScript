//BrainJS and MNIST imports.
const fs = require('fs');
const brain = require('brain.js');
const mnist = require('./mnist');

//Finding the max value in the data and parsing its key into an array of
//10 numbers.

function maxScore(obj) {
    let maxKey = 0;
    let maxValue = 0;

    Object.entries(obj).forEach(entry => {
        let key = entry[0];
        let value = entry[1];
        if (value > maxValue) {
            maxValue = value;
            maxKey = parseInt(key, 10);
        }
    });
    
    return maxKey;
}

async function run() {

    //Taking the model in JSON form from train.js and converting back into its
    //BrainJS model form.

    const networkModel = JSON.parse(fs.readFileSync('./data/model.json', 'utf8'));

    const net = new brain.NeuralNetwork();
    net.fromJSON(networkModel);

    //Getting the images from mnist.js and converting them into binary, just as in
    //train.js.

    const testDigits = await mnist.getTestImages();
    const testLabels = await mnist.getTestLabels();

    const imageSize = 28 * 28;
    let errors = 0;

    for (let ix = 0; ix < testLabels.length; ix++) {
        const start = ix * imageSize;
        const input = testDigits.slice(start, start + imageSize).map(mnist.normalize);

        //Finding the max value from the model and checking that it matches the predicted
        //value. If it does not, the model error is increased.

        const detection = net.run(input);        
        const max = maxScore(detection);
        //console.log(max, testLabels[ix]);
        if (max !== testLabels[ix]) {
            errors++;
        }
    }

    //Printing the model.

    console.log(`Total           : ${testLabels.length}`);
    console.log(`Number of errors: ${errors}`);
    console.log(`Accuracy:       : ${(testLabels.length - errors) * 100 / testLabels.length} %`);
}

run();