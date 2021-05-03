# NLP Sentiment Analysis



## Data
Two datasets were considered:
- Sentiment140 (in-domain training and test data)
- Dublin City Council data (out-of-domain test data)

## Models
Logistic Regression and [BERT](https://huggingface.co/bert-base-uncased) models were trained on 110,000 training examples of the `Sentiment140-train` dataset and evaluated on `Sentiment-140-valid` (10% or 11,000 examples from the training set), `Sentiment140-test`, and `Dublin City Council` datasets.


## Training

### 1. Baseline
Before using Machine Learning at all, I used a simple dummy approach to see the minimum possible accuracy that I can get. Logic: For each word in the corpus, 2 different scores were given: How many times this word appeared in negative tweets, and how many times in positive tweets. To make prediction, I simply summed the negative and the positive scores across each word >>> If negative sum is higher, predict `negative` and vice versa. This approach gave 65% accuracy on the unseen training data, so I knew that this is the minimum that I need to aim for on the validation set.

### 2. Logistic Regression
To convert text into numbers, I used TF-IDF for text vectorization. Only 1,000 maximally relevant tokens/words were kept. Different hyperparameters were tested with `Grid Search` on a small subset of training examples and later used in training the model on more data. To use the model as a 3-label-classifier, the prediction probabilities were compared against a threshold , the predictions were set to `neutral`.

### 3. BERT
Pretrained [BERT model](https://huggingface.co/bert-base-uncased) in combination with pretrained tokenizer (both are `bert-base-uncased`) was used as the final model. I used a dropout layer with probability of 30% as regularization and a softmax layer on top of the model's encoder to convert the pooled output into labels `0` and `1`. This architecture was chosen because it is recommended in the [original research paper](https://arxiv.org/abs/1810.04805). BERT was fine-tuned during 10 epochs and the parameters that proved the best accuracy on the validation set were saved. The same approach was used to convert BERT into 3-label-classifier as with LogReg.

**Accuracy results**:

![image](/img/accuracy_results.png)



**Accuracy evolution** - *BERT model seems to be overfitting*:

![image](/img/acc_evol.png)

**Loss evolution** - *BERT model seems to be overfitting*:

![image](/img/loss_evol.png)


**Main challenges**:
- Training data contains *2 labels* (`negative/positive`) while the test data contains *3* (`negative/positive/neutral`). The accuracy drops dramatically from binary to 3-label-classification.
- Dublin City Council is out-of-domain data and differs notably from the training data, especially in the formality of the language.

**Decisions taken along the process**:
- Since the training set only contained labels `0` and `4`, labels `4` were converted into `1` for convenience.
- To convert the model from binary into 3-label-classifier, the prediction probabilities were tested against a `threshold`. If the model's prediction certainty/probability is low (i.e. the model is not very confident about its prediction), the prediction was set to `2` or `neutral`. This improved the accuracy by +2.2% on the final test set for Logistic Regression but almost did not improve it for BERT (+0.03%).
- To understand why the models made errors, I found top losses for each test example (both Sentiment140 and Dublin City council data), i.e. cases when the model was very confident in its prediction but it was incorrect. 
- Additionally, in case of Logistic Regression, for each token in top loss examples, I printed the model coefficients to see how the token impacted the final score.
- I used a dropout layer (`p=30%`) as regularization and a linear + softmax layer on top of the BERT's encoder to convert the raw pooled output into labels `0` and `1` (recommended by the original research paper).


**Error analysis and interesting observations**:
- Either the annotation quality is below desired or the model has difficulties recognizing **ironic/sarcastic** sentences. Example - Tweet id2327969640: 
```
`@personalhudgens aww i bet i'm good thank you x` seems to convey a positive sentiment while carrying a negative label.
```
- Emojis could help to recognize sarcastic tweets if they were preprocessed properly. Examples:
```
Correct label: 0 | Prediction: 1
Tokenized text: @kirstiealley my dentist is great (0.15) but she 's (-0.01) expensive … (-0.07) =(
```
```
Correct label: 0 | Prediction: 1
Tokenized text: xxmaj night (0.02) at the xxmaj museum tonite instead (-0.0) of xxup up . (-0.03) :( oh (-0.04) well . (-0.03) that 4 (-0.03) yr old (-0.0) better (0.06) enjoy (0.09) it . (-0.03) xxup lol (0.1)
```
```
Correct label: 1 | Prediction: 0
Tokenized text: xxmaj class (-0.02) … (-0.07) xxmaj the 50d is supposed (-0.04) to come (-0.01) today (-0.03) :)
```
- Many emoticons mean the same thing but are not recognized as such. Like `=)`, `:-)` and `:)`. Other emoticons are not recognized at all (like ❤️ ) while they carry a lot of meaning.
- In case of logistic regression, certain numbers have negative sentiment which does not really make sense.
- Both models' accuracy was **lower** on topics of `Mobility`, `Health`, `Safety`, `Housing`, `Infrastructure` for Dublin City Council test set, and **higher** for `Public Spaces`, `Tourism and Hospitality` and `Community and Culture`. This might because Dublin's spheres of mobility, infrastructure and housing are very special to the city and cannot be generalized well from data from other places.


**Model's improvement suggestions**:
- First and foremost, the model should be trained on data with 3 labels, not with 2. It might be most helpful to hire some freelancers to annotate `neutral` tweets to get high quality data.
- Edit the preprocessing steps. A code example of the implementation of these improvements is in the end of the notebook (`Research questions` section):
  - Replace hyperlinks with special tokens (like `xxhyperlink`).
  - Replace all the different numbers with a single token (like `xxnumber`).
  - Preprocess emojis and convert other html symbols into recognizable tokens (Convert `:)`, `=)`, `:-)`, etc. into the same token `xxsmileemoji` and `&#10084;` and `<3` into `xxheart`).
  - It would be probably better to remove the remaining punctuation altogether.
- To address the domain differences between Sentiment140 and Dublin City Council datasets, a self-supervised [language model](https://www.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html) (that predicts the next word in the sentence) could be trained on unlabeled tweets scraped specifically from Dublin area. This model could be used as the pretrained model for classification later, instead of the original pretrained `BERT-base-uncased`. This technique has proved itself quite well in similar classification tasks.
- The results would improve slightly if it was trained on more training examples. I stopped with 110,000 because with more it would have taken more time than was possible to use Google Colab. But from my observations, the results increased by only 1-2% by going from 10,000 to 110,000 examples.
- Spend more time on tweaking the models' hyperparameters. For Logistic Regression, `GridSearchCV` with a small set of parameters was used to find the best ones on 200 training examples (otherwise, it took too long with 1,000 features). More data could be used as well as bigger set of parameters tested. For BERT, different dropout rates as well as learning rates could be tested.

**Author**: [Pavlo Seimskyi](mailto:pavlo@dataforfuture.org)
