class BPE_Tokenizer {
    constructor() {
        this.vocab = new Set();
        this.tokenToIndex = {};
        this.indexToToken = {};
    }

    tokenize(text) {
        let tokens = [];
        let words = text.split(/\s+/);
        for (let word of words) {
            let subwords = this.getSubwords(word + '</w>');
            for (let sw of subwords) {
                tokens.push(this.tokenToIndex[sw] ?? this.tokenToIndex['</u>']);
            }
        }
        return tokens;
    }

    getSubwords(word) {
        let subwords = [];
        while (word) {
            let subword = this.findLongestSubword(word);
            if (!subword) {
                subwords.push('</u>');
                break;
            }
            subwords.push(subword);
            word = word.slice(subword.length);
        }
        return subwords;
    }

    findLongestSubword(word) {
        for (let i = word.length; i > 0; i--) {
            let possibleSubword = word.slice(0, i);
            if (this.vocab.has(possibleSubword)) {
                return possibleSubword;
            }
        }
        return null;
    }

    detokenize(tokenIds) {
        let words = [];
        let currentWord = '';
        for (let token_id of tokenIds) {
            let token = this.indexToToken[token_id] ?? '</u>';
            if (token === '</w>') {
                words.push(currentWord);
                currentWord = '';
            } else {
                currentWord += token;
            }
        }
        if (currentWord) {
            words.push(currentWord);
        }
        return words.join(' ').replace(/<\/w>/g, ' ').replace(/<\/u>/g, '');
    }

    loadVocab(tokenToIndex) {
        this.tokenToIndex = tokenToIndex;
        this.indexToToken = {};
        for (let [token, index] of Object.entries(tokenToIndex)) {
            this.indexToToken[index] = token;
        }
        this.vocab = new Set(Object.keys(tokenToIndex));
    }

    getVocabSize() {
        return Object.keys(this.tokenToIndex).length;
    }
}

// Example Usage
// let tokenizer = new BPE_Tokenizer();
// tokenizer.loadVocab(yourTokenToIndexMapping); // Replace with your mapping

// let tokenIds = tokenizer.tokenize("Example text to tokenize");
// console.log("Token IDs:", tokenIds);

// let detokenizedText = tokenizer.detokenize(tokenIds);
// console.log("Detokenized Text:", detokenizedText);
