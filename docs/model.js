BLOCK_SIZE = 64;


async function generateWithOnnx(sess, inputName, labelName, prompt, maxLen = 100, temperature = 0.5, repetition = 1.1) {
    let inputIds = tokenizer.tokenize(prompt);
    let generated = [];

    for (let i = 0; i < maxLen; i++) {
        // Crop input to block size
        let idxCond = (inputIds.length <= BLOCK_SIZE) ? inputIds : inputIds.slice(-BLOCK_SIZE);
        // Convert each number in idxCond to a BigInt
        const bigIntIdxCond = idxCond.map(id => BigInt(id));
        // Convert to tensor
        const inputTensor = new ort.Tensor('int64', new BigInt64Array(bigIntIdxCond), [1, bigIntIdxCond.length]);
        const logits = await sess.run({[inputName]: inputTensor});
        // console.log(logits);
        let probs = softmaxAndTemperature(logits[labelName].data, temperature);

        // Apply repetition penalty
        probs = applyRepetitionPenalty(probs, inputIds, repetition);

        // Sample from the distribution
        const idxNext = multinomialSampling(probs, 1);
        inputIds.push(idxNext);
        generated.push(idxNext);
    }

    return generated;
}

function softmaxAndTemperature(logits, temperature) {
    const adjustedLogits = logits.map(l => Math.exp(l / temperature));
    const sum = adjustedLogits.reduce((a, b) => a + b, 0);
    return adjustedLogits.map(value => value / sum);
}

function applyRepetitionPenalty(probs, inputIds, repetition) {
    let counts = {};
    inputIds.forEach(id => counts[id] = (counts[id] || 0) + 1);

    for (let i = 0; i < probs.length; i++) {
        if (counts[i]) {
            probs[i] /= Math.pow(repetition, counts[i]);
        }
    }

    return probs;
}


function multinomialSampling(probs, numSamples) {
    let sample = [];
    for (let i = 0; i < numSamples; i++) {
        let cumSum = 0;
        const r = Math.random();
        for (let j = 0; j < probs.length; j++) {
            cumSum += probs[j];
            if (r <= cumSum) {
                sample.push(j);
                break;
            }
        }
    }
    return sample.length === 1 ? sample[0] : sample;
}

async function speakOnnx(sess, prompt, maxLen = 100, temperature = 0.5, repetition = 1) {
    const inputName = model.inputNames[0];
    const labelName = model.outputNames[0];
    const generated = await generateWithOnnx(sess, inputName, labelName, prompt, maxLen, temperature, repetition);
    console.log(prompt + ' ' + tokenizer.detokenize(generated));
}

async function generateNextTokenWithModel(inputIds) {
    const temperature = 0.4;
    const repetition = 1.01;
    const inputName = model.inputNames[0];
    const labelName = model.outputNames[0];

    // Crop input to block size
    let idxCond = (inputIds.length <= BLOCK_SIZE) ? inputIds : inputIds.slice(-BLOCK_SIZE);
    // Convert each number in idxCond to a BigInt
    // const bigIntIdxCond = idxCond.map(id => BigInt(id));
    // Convert to tensor
    // const inputTensor = new ort.Tensor('int64', new BigInt64Array(bigIntIdxCond), [1, bigIntIdxCond.length]);
    const inputTensor = new ort.Tensor('int32', new Int32Array(idxCond), [1, idxCond.length]);
    const logits = await model.run({[inputName]: inputTensor});
    console.log(logits);

    let probs = softmaxAndTemperature(logits[labelName].data, temperature);

    // Apply repetition penalty
    probs = applyRepetitionPenalty(probs, inputIds, repetition);

    // Sample from the distribution
    const idxNext = multinomialSampling(probs, 1);

    return idxNext;
}