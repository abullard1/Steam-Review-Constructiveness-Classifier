<p align="center">
  <img src="https://i.ibb.co/Ky0wcYy/abullard1-steam-review-constructiveness-classifier-logo-modified-1.png" alt="Logo" width="200">
</p>

<h1 align="center">Fine-tuned ALBERT Model for Constructiveness Detection in Steam Reviews</h1>

## Repository Contents
1. **ALBERT-V2 Steam Game-Review Constructiveness Classification Model**

2. **1.5k Steam Reviews with Binary Constructiveness Labels**

3. **Jupyter Notebooks for: Data Filtering, Data Preprocessing, Training, Inference, Evaluation**


## Model Summary

The model contained in this repository is a fine-tuned version of **albert-base-v2**, designed to classify whether Steam game reviews are constructive or non-constructive. It was trained on the [1.5K Steam Reviews Binary Labeled for Constructiveness](https://huggingface.co/datasets/abullard1/steam-reviews-constructiveness-binary-label-annotations-1.5k) dataset, also contained in this repository.

### Intended Use

The model can be applied in any scenario where it's important to distinguish between helpful and unhelpful textual feedback, particularly in the context of gaming communities or online reviews. Potential use cases are platforms like **Steam**, **Discord**, or any community-driven feedback systems where understanding the quality of feedback is critical.

### Limitations

- **Domain Specificity**: The model was trained on Steam reviews and may not generalize well outside gaming.
- **Dataset Imbalance**: The training data has an approximate **63.04%-36.96%** split between non-constructive and constructive reviews.

<br>

## Dataset

### Dataset Summary

The dataset contained in this repository consists of **1,461 Steam reviews** from **10 of the most reviewed games** in the base [100 Million+ Steam Reviews](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) dataset. Each game has approximately the same number of reviews. Each review is annotated with a **binary label** indicating whether the review is **constructive** or not.

Also available as additional data are **train/dev/test split** CSV files. These contain the features of the base dataset, concatenated into strings, next to the binary constructiveness labels. These CSVs were used to train the model.

The dataset is designed to support tasks related to **text classification**, particularly **constructiveness detection** in the gaming domain. It is particularly useful for training models like **BERT** and its derivatives or any other NLP models aimed at classifying text for constructiveness.

### Dataset Structure

The dataset contains the following columns:

- **id**: A unique identifier for each review.
- **game**: The name of the game being reviewed.
- **review**: The text of the Steam review.
- **author_playtime_at_review**: The number of hours the author had played the game at the time of writing the review.
- **voted_up**: Whether the user marked the review/the game as positive (True) or negative (False).
- **votes_up**: The number of upvotes the review received from other users.
- **votes_funny**: The number of "funny" votes the review received from other users.
- **constructive**: A binary label indicating whether the review was constructive (1) or not (0).

#### Example Data

| id   | game                | review                                                                | author_playtime_at_review | voted_up | votes_up | votes_funny | constructive |
|------|---------------------|-----------------------------------------------------------------------|---------------------------|----------|----------|-------------|--------------|
| 1024 | Team Fortress 2     | shoot enemy                                                           | 639                       | True     | 1        | 0           | 0            |
| 652  | Grand Theft Auto V  | 6 damn years and it's still rocking like its g...                     | 145                       | True     | 0        | 0           | 0            |
| 1244 | Terraria            | Great game highly recommend for people who like...                    | 569                       | True     | 0        | 0           | 1            |
| 15   | Among Us            | So good. Amazing game of teamwork and betrayal...                     | 5                         | True     | 0        | 0           | 1            |
| 584  | Garry's Mod         | Jbmod is trash!!!                                                     | 65                        | True     | 0        | 0           | 0            |

### Labeling Criteria

- **Constructive (1)**: Reviews that provide helpful feedback, suggestions for improvement, constructive criticism, or detailed insights into the game.
- **Non-constructive (0)**: Reviews that do not offer useful feedback, lack substance, are vague, off-topic, irrelevant, or trolling.

Please note that the **dataset is unbalanced**: **63.04%** of the reviews were labeled as non-constructive, while **36.96%** were labeled as constructive. Please take this into account when utilizing the dataset.

### Data Preparation

The dataset features were combined into a single string per review, formatted as follows:

**`Review: {review}, Playtime: {author_playtime_at_review}, Voted Up: {voted_up}, Upvotes: {votes_up}, Votes Funny: {votes_funny}"`**

and then fed to the model accompanied by the respective **constructive** labels.

This approach of concatenating the features into a simple string offers a good **trade-off** between **complexity** and **performance**.

<br>

## Code / Jupyter Notebooks

Originally, the model was created as part of the process of evaluating several different models against eachother. These models were: [BERT](https://huggingface.co/google-bert/bert-base-uncased), [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), [ALBERT-V2](https://huggingface.co/albert/albert-base-v2), [XLNet](https://huggingface.co/xlnet/xlnet-base-cased), [GPT-2](https://huggingface.co/openai-community/gpt2)

The repository contains the following **five jupyter notebooks**:

- **Filtering**: Filtering and reduction of the base dataset to a smaller more usable one.
- **Preprocessing**: Basic and conservative preprocssing, fit for transformer-based LLM fine-tuning. Simple statistical analysis of the dataset and annotations.
- **Training**: Fine-Tuning / Training of the model.
- **Inference**: Simple testing environment.
- **Evaluation**: Evaluation environment to evaluate the aformentioned classification models against eachother.

<u>Note:</u>
Please take into account that the jupyter notebooks are a mix of working with [Google Colab](https://colab.research.google.com/) computing resources and local resources.
Therefore, in order to use them, they **need to be modified** to match your own personal working environment.

<br>

## Evaluation Results

The model was trained and evaluated using a **80/10/10 Train/Dev/Test** split, achieving the following performance metrics on the test set:

- **Accuracy**: 0.80
- **Precision**: 0.80
- **Recall**: 0.82
- **F1-score**: 0.79

These results indicate that the model performs reasonably well at identifying the correct label (~80%).

<br>

## How to Use

### Hugging Face Space

Explore and test the model interactively on its [Hugging Face Space](https://huggingface.co/spaces/abullard1/steam-review-constructiveness-classifier).

### Transformers Library

To use the model programmatically, use this Python snippet:

```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
torch_d_type = torch.float16 if torch.cuda.is_available() else torch.float32

base_model_name = "albert-base-v2"

finetuned_model_name = "abullard1/albert-v2-steam-review-constructiveness-classifier"

classifier = pipeline(
    task="text-classification",
    model=finetuned_model_name,
    tokenizer=base_model_name,
    device=device,
    top_k=None,
    truncation=True,
    max_length=512,
    torch_dtype=torch_d_type
)

review = "Review: I think this is a great game but it still has some room for improvement., Playtime: 12, Voted Up: True, Upvotes: 1, Votes Funny: 0"
result = classifier(review)
print(result)
```

### Jupyter Notebooks in Repository

Alterantively the **Jupyter Notebooks** contained in this repository can be used to test the model or even replicate the process of training/fine-tuning.

The notebooks contain **useful code comments** throughout, describing what is happening at every step of the way.

*An interesting case for further modifications or improvements, would be to **augment or modify the training dataset**. Feel free to do so.*

## <u>License</u>

The model, dataset and code in this repository is licensed under the *[MIT License](https://mit-license.org/)*, allowing open and flexible use of the dataset for both academic and commercial purposes.
