async function displayAscii() {
    initModel();
    initTokeniser();
    const response = await fetch('ascii.txt');
    const text = await response.text();
    const lines = text.split('\n');

    for (let i = 0; i < lines.length; i++) {
        setTimeout(() => {
            document.getElementById('content').innerText += lines[i] + '\n';
        }, 50 * i);
    }

    setTimeout(() => {
        displayGeneratedText();
        // speakOnnx(model, "Hi there! I'm Nicolas STAS ", 100, 0.9, 1.1);
    }, 50 * lines.length);
}

function displayHelloWorld() {
    const helloBox = document.getElementById('helloBox');
    helloBox.classList.remove('hidden'); // Make sure this correctly shows the box
    helloBox.value = text; // Clear the box

    setInterval(() => {
        helloBox.value += 'Hello World!\n';
    }, 500);
}
let model;

let tokenizer = new BPE_Tokenizer();

async function initTokeniser() {
    let response = await fetch('latest.openweb+Nicolas.json');
    let TokenToIndex = await response.json();
    tokenizer.loadVocab(TokenToIndex);

    // console.log("TokenToIndex:", TokenToIndex);
    // const text = "Hi there! I'm Nicolas Stas , a tech enthusiast and creative technology at ESILV Paris Graduate School of Engineering . Lets make something amazing ! Hi there ! I'm Nicolas Stas , a tech-savvy mind at the DeVinciBot robotics clubs technology at the DeVinciBot robotics club , where I'm planting for the realms of advanced drones and precision . My journey-delved into your work without the tech-savvy skills of engineering . Lets just forgin . Arhowarly wither !" ;
    // let tokenIds = tokenizer.tokenize(text);
    // console.log("Token IDs:", tokenIds);

    // let detokenizedText = tokenizer.detokenize(tokenIds);
    // console.log("Detokenized Text:", detokenizedText);
}


// Initialize ONNX model
async function initModel() {
    
    // model = new ort.InferenceSession();
    // model = await ort.InferenceSession
    //                       .create('latest.llm.quant.onnx', 
    //                       { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    model = await ort.InferenceSession
                          .create('latest.llm.quant.onnx');
    console.log("Model Loaded");
    console.log(model);
}


async function displayGeneratedText() {
    const helloBox = document.getElementById('helloBox');
    helloBox.classList.remove('hidden'); 
    let currentText = "Hi there! I'm Nicolas STAS ";
    helloBox.value = currentText; // Initial prompt

    // check if the model is loaded
    if (!model) {
        console.log("Model not loaded");
        return;
    }

    let currentTokens = tokenizer.tokenize(currentText); //initialTokens
    const maxTokens = 10000; // Set a limit to the number of tokens to generate

    for (let i = 0; i < maxTokens; i++) {
        const nextToken = await generateNextTokenWithModel(currentTokens);
        //  add the new token to the current text
        currentTokens.push(nextToken);
        currentText = tokenizer.detokenize(currentTokens);
        // replace "." with ".\n" for readability
        currentText = currentText.replaceAll(". ", ".\n");
        // currentText += nextWord;
        helloBox.value = currentText;

        await new Promise(resolve => setTimeout(resolve, 10)); // Delay for readability
        if (currentText.length > 2048) {
            currentText = "Hi there! I'm Nicolas STAS ";
            helloBox.value = currentText;
            currentTokens = tokenizer.tokenize(currentText);
        }
    }
}


window.onload = displayAscii;
