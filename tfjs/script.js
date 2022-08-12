var values = [];
var tensorData;
var model;

function setUi() {
    $('.ui.radio.checkbox').checkbox();
}

async function generate_data() {
    var input_numbers = $('#tds').val();

    var equation = $('input[name="equation"]:checked').val();

    values = [];
    for (i = 0; i <= input_numbers; i++) {
        var x = 0;
        if (equation == 1) {
            x = (2 * i + 3) * 3;
        }
        else if (equation == 4) {
            x = (((i / 2) ^ 2) + 10) * 2;
        }
        else if (equation == 3) {
            x = i + 4;
        }
        else if (equation == 2) {
            x = 2 * i;
        }

        values.push(
            {
                x: i,
                y: x,
            }
        )
    }


    tfvis.render.scatterplot(
        { name: 'X v/s Y' },
        { values },
        {
            xLabel: 'X',
            yLabel: 'Y',
            height: 300
        }
    );

    $('.gm').removeClass("hide");

}

function generatemodel() {
    model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);
    tensorData = convertToTensor(values);
    $('.tm').removeClass("hide");

}

async function trainmodel() {
    const { inputs, labels } = tensorData;
    await trainModel(model, inputs, labels);
    console.log('Done Training');
    $('.tem').removeClass("hide");

}

function testmodel() {
    testModel(model, values, tensorData);
}

function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.x)
        const labels = data.map(d => d.y);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));


    return model;
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;
    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));
        const unNormXs = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);
        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] }
    });

    const originalPoints = inputData.map(d => ({
        x: d.x, y: d.y,
    }));


    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data' },
        { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
        {
            xLabel: 'X',
            yLabel: 'Y',
            height: 300
        }
    );
}

document.addEventListener('DOMContentLoaded', setUi);
