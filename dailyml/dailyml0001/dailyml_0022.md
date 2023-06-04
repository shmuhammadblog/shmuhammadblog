---
title: "Active Learning"
date: '2022-10-15'
output: html_document
---
# Maximum performance

Claire and Phil aren't on the same page with their plan.

They want to train a machine learning model but want to minimize the number of samples they need to label. Labeling takes too long, and they want to avoid it as much as possible.

Claire argues that they don't need to train with the entire dataset. Instead, she believes they can maximize the model's performance without using all the data.

Phil disagrees. He argues that the only way to achieve the maximum possible performance is to train with the entire dataset. Since they aren't willing to label all the data, they will need to settle for a mediocre model.

What's your opinion about this situation?

- Achieving the maximum possible performance without using the entire dataset is theoretically possible but very unlikely.

- They can achieve the maximum possible performance without using the entire dataset by randomly sampling a portion of the data, labeling it, and training the model.

- They can achieve the maximum possible performance without using the entire dataset, but they need a good strategy to sample the data they will label to train the model.

- They will never achieve the maximum possible performance without using the entire dataset.

## Answer

> With active learning, we can build a model that will achieve better performance with fewer labeled samples by allowing the algorithm to choose the data that will provide the most information to its training process.


Claire and Phil do not need to use the entire dataset to build a model that reaches its maximum possible performance. However, they will need a smart strategy to select the data they need to label.

Let's imagine a dataset with two classes that we can represent in two dimensions and a linear model that splits the data into two groups. Any samples around the lines' boundaries that separate both classes are critical in our dataset. Those samples help the model decide how to split the data!

But what about samples far away from the split? They contribute much less to the model, and we don't need them to find the separation between classes. The same happens with duplicate samples or samples that are too similar to existing ones.

Claire and Phil, however, can't depend on randomly sampling the dataset to decide which instances to label. They need a better strategy to determine which samples to pick.

This scenario is an example of Active learning. This learning technique allows us to build a better-performing machine learning model using fewer training labels by strategically choosing the samples to train the model.


## Recommended reading

1. [Active learning](https://articles.bnomial.com/active-learning)
2. [A Short Introduction to Active Learning](https://medium.com/cognifeed/https-medium-com-cognifeed-a-short-introduction-to-active-learning-8c583c81f4e5)
3. [Tutorial on Active learning](https://hunch.net/~active_learning/active_learning_icml09.pdf)

<style>
body {
text-align: justify}
</style>
