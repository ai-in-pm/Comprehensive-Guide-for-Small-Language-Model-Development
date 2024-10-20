import React from 'react';
import { Sliders } from 'lucide-react';

const EvaluationAndFineTuning: React.FC<{ isActive: boolean }> = ({ isActive }) => {
  if (!isActive) return null;

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 flex items-center">
        <Sliders className="mr-2" />
        Evaluation and Fine-tuning
      </h2>
      <p className="text-gray-600">
        Proper evaluation and fine-tuning are crucial for optimizing the performance of Small Language Models (SLMs). This section covers key strategies and techniques for assessing and improving your model.
      </p>
      
      <h3 className="text-2xl font-semibold text-gray-700">1. Assess model performance</h3>
      <p className="text-gray-600">
        Evaluate your SLM on validation and test sets using appropriate metrics. For language models, common metrics include:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Perplexity: Measures how well the model predicts a sample</li>
        <li>BLEU score: Evaluates the quality of machine-translated text</li>
        <li>ROUGE score: Assesses the quality of summarization tasks</li>
        <li>F1 score: Useful for classification tasks</li>
        <li>Task-specific metrics: Tailored to your SLM's specific use case</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">2. Implement fine-tuning techniques</h3>
      <p className="text-gray-600">
        Use transfer learning and few-shot learning techniques to adapt your SLM to specific tasks or domains:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Gradual unfreezing: Slowly unfreeze layers during fine-tuning</li>
        <li>Discriminative fine-tuning: Use different learning rates for different layers</li>
        <li>ULMFiT: Utilize language model fine-tuning techniques</li>
        <li>Few-shot learning: Train on a small number of examples for new tasks</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">3. Optimize hyperparameters</h3>
      <p className="text-gray-600">
        Fine-tune hyperparameters based on evaluation results. Consider these techniques:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Grid search: Exhaustive search through a specified parameter space</li>
        <li>Random search: Randomly sample from the parameter space</li>
        <li>Bayesian optimization: Use probabilistic model to select the most promising hyperparameters</li>
        <li>Population-based training: Evolve a population of models and hyperparameters</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">4. Conduct ablation studies</h3>
      <p className="text-gray-600">
        Perform ablation studies to understand the importance of different components:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Remove or modify individual layers or components</li>
        <li>Vary the size of embedding dimensions or hidden layers</li>
        <li>Experiment with different activation functions</li>
        <li>Analyze the impact of various regularization techniques</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">5. Perform cross-validation</h3>
      <p className="text-gray-600">
        Use k-fold cross-validation for robust performance estimation:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Implement stratified k-fold for imbalanced datasets</li>
        <li>Use time series cross-validation for sequential data</li>
        <li>Consider nested cross-validation for hyperparameter tuning</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">6. Develop domain-specific evaluation metrics</h3>
      <p className="text-gray-600">
        Create evaluation metrics tailored to your SLM's specific use case:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Task completion rate for goal-oriented dialogues</li>
        <li>Domain-specific accuracy measures (e.g., medical term accuracy)</li>
        <li>User satisfaction scores for interactive applications</li>
        <li>Custom metrics that combine multiple performance aspects</li>
      </ul>
      
      <h3 className="text-2xl font-semibold text-gray-700">7. Implement continuous evaluation pipelines</h3>
      <p className="text-gray-600">
        Set up automated evaluation pipelines to continuously monitor your SLM:
      </p>
      <ul className="list-disc list-inside text-gray-600 ml-4">
        <li>Integrate evaluation into your CI/CD pipeline</li>
        <li>Implement A/B testing for comparing model versions</li>
        <li>Set up alerting systems for performance degradation</li>
        <li>Use version control for models and evaluation results</li>
      </ul>
      
      <div className="mt-8 p-6 bg-blue-50 rounded-lg border border-blue-200">
        <h4 className="text-xl font-semibold text-blue-800 mb-4">Code Snippet: Advanced Fine-tuning with ULMFiT</h4>
        <pre className="bg-gray-800 text-white p-4 rounded-lg overflow-x-auto">
          <code>{`
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class ULMFiTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = 0

    def training_step(self, model, inputs):
        if self.current_epoch < 1:
            # Freeze all but the last layer
            for param in model.base_model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif self.current_epoch < 2:
            # Unfreeze the last layer of the base model
            for param in model.base_model.encoder.layer[-1].parameters():
                param.requires_grad = True
        else:
            # Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True

        return super().training_step(model, inputs)

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        self.current_epoch += 1
        return result

def fine_tune_evaluate_slm(model_name, dataset_name, num_labels):
    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and preprocess dataset
    dataset = load_dataset(dataset_name)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
    )

    # Initialize ULMFiTTrainer
    trainer = ULMFiTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    return model, tokenizer, eval_results

# Usage
model_name = "distilbert-base-uncased"  # A smaller BERT model suitable for SLMs
dataset_name = "imdb"  # Example dataset
num_labels = 2  # Binary classification for sentiment analysis

fine_tuned_model, tokenizer, eval_results = fine_tune_evaluate_slm(model_name, dataset_name, num_labels)
          `}</code>
        </pre>
      </div>
      
      <div className="mt-8 p-6 bg-green-50 rounded-lg border border-green-200">
        <h4 className="text-xl font-semibold text-green-800 mb-4">Best Practices for SLM Evaluation and Fine-tuning</h4>
        <ul className="list-disc list-inside text-green-700 space-y-2">
          <li>Use a diverse set of evaluation metrics to get a comprehensive view of model performance</li>
          <li>Implement early stopping during fine-tuning to prevent overfitting</li>
          <li>Experiment with different learning rates and schedules for fine-tuning</li>
          <li>Use techniques like gradual unfreezing when fine-tuning pre-trained models</li>
          <li>Regularly evaluate on held-out test sets to ensure generalization</li>
          <li>Consider using techniques like mixout for better fine-tuning of small models</li>
          <li>Implement error analysis to identify areas for improvement</li>
          <li>Use data augmentation techniques to increase the diversity of your training data</li>
          <li>Implement ensemble methods to combine multiple fine-tuned models for improved performance</li>
          <li>Utilize techniques like knowledge distillation to transfer knowledge from larger models to your SLM</li>
        </ul>
      </div>
      
      <div className="mt-8 p-6 bg-yellow-50 rounded-lg border border-yellow-200">
        <h4 className="text-xl font-semibold text-yellow-800 mb-4">Challenges and Strategies for SLM Evaluation and Fine-tuning</h4>
        <ul className="list-disc list-inside text-yellow-700 space-y-2">
          <li>Limited data for fine-tuning: Use few-shot learning techniques and data augmentation</li>
          <li>Overfitting during fine-tuning: Implement regularization, early stopping, and gradual unfreezing</li>
          <li>Domain adaptation: Employ domain-specific pre-training or progressive transfer learning</li>
          <li>Balancing performance across tasks: Use multi-task learning approaches and careful task weighting</li>
          <li>Efficient evaluation: Implement progressive evaluation techniques and smart sampling strategies</li>
          <li>Handling concept drift: Implement continuous learning and periodic re-evaluation</li>
          <li>Resource constraints: Use quantization-aware fine-tuning and efficient architecture search</li>
          <li>Interpretability: Implement attention visualization and layer-wise relevance propagation</li>
          <li>Bias mitigation: Use adversarial debiasing techniques and fairness-aware fine-tuning</li>
          <li>Robustness to adversarial examples: Incorporate adversarial training in the fine-tuning process</li>
        </ul>
      </div>
    </div>
  );
};

export default EvaluationAndFineTuning;