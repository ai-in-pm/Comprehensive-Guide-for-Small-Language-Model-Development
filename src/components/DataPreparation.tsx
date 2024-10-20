import React from 'react';
import { Database } from 'lucide-react';

const DataPreparation: React.FC<{ isActive: boolean }> = ({ isActive }) => {
  if (!isActive) return null;

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 flex items-center">
        <Database className="mr-2" />
        Data Preparation
      </h2>
      <p className="text-gray-600">
        Proper data preparation is crucial for training an effective Small Language Model (SLM). This section covers the key steps and considerations in preparing your dataset.
      </p>
      
      <h3 className="text-2xl font-semibold text-gray-700">1. Select or create a domain-specific dataset</h3>
      <p className="text-gray-600">
        The selection or creation of a domain-specific dataset is a critical first step in developing your SLM. For CompactLM, we've chosen a combination of publicly available datasets: Wikipedia articles for general knowledge, OpenSubtitles for conversational data, and the CNN/Daily Mail dataset for summarization tasks. This diverse mix allows our model to learn from a wide range of text styles and content, supporting its general-purpose nature while still providing specialized data for key tasks like summarization.
      </p>
      <p className="text-gray-600">
        When selecting your dataset, consider the specific requirements of your SLM's target domain and use cases. If your model is intended for a specialized field, such as medical text analysis or legal document processing, you'll need to source or create datasets that accurately represent the language and concepts in those domains. This might involve partnering with domain experts or organizations to access or annotate relevant data.
      </p>
      <p className="text-gray-600">
        It's also important to consider the quality and diversity of your data. High-quality, diverse data is crucial for training a robust model. This means ensuring your dataset covers a wide range of topics, writing styles, and linguistic phenomena relevant to your target domain. For instance, if your SLM is intended for use in customer service chatbots, your dataset should include a variety of customer queries, different tones (formal, casual, frustrated), and diverse vocabulary related to customer service scenarios.
      </p>
      <p className="text-gray-600">
        Lastly, consider the size of your dataset. While SLMs generally work with smaller datasets compared to their larger counterparts, you still need sufficient data to train an effective model. The exact amount will depend on the complexity of your task and the size of your model. Start with a reasonable baseline (e.g., a few gigabytes of text for a general-purpose model) and be prepared to iterate, potentially expanding your dataset if you find the model's performance lacking in certain areas.
      </p>
      
      <h3 className="text-2xl font-semibold text-gray-700">2. Preprocess and clean the data</h3>
      <p className="text-gray-600">
        Preprocessing and cleaning your data is a crucial step that can significantly impact the quality and effectiveness of your SLM. This process involves several key steps, each aimed at improving the quality and consistency of your training data. For CompactLM, we've implemented a comprehensive preprocessing pipeline that includes lowercasing, special character removal, tokenization, and stopword removal.
      </p>
      <p className="text-gray-600">
        The first step in preprocessing is often text normalization. This includes converting all text to lowercase (unless case is semantically important in your domain), handling special characters, and standardizing formatting. For instance, you might choose to remove or replace certain punctuation marks, convert numbers to a standard format, or expand common abbreviations. The goal is to reduce noise in the data and ensure consistency, which helps the model learn more effectively.
      </p>
      <p className="text-gray-600">
        Tokenization is another critical step in the preprocessing pipeline. This involves breaking down the text into individual tokens (usually words or subwords) that the model can process. The choice of tokenization method can significantly impact your model's performance and efficiency. For SLMs, which often have limited vocabulary sizes, subword tokenization methods like Byte-Pair Encoding (BPE) or WordPiece can be particularly effective. These methods allow the model to handle out-of-vocabulary words by breaking them down into familiar subword units.
      </p>
      <p className="text-gray-600">
        Finally, consider domain-specific preprocessing steps that might be relevant to your SLM's target application. For instance, if your model will be working with social media data, you might need to handle hashtags, @mentions, or emojis in a specific way. If it's dealing with technical or scientific text, you might need to preserve certain formatting or special characters that carry semantic meaning. Always balance the need for cleaning and standardization with the importance of preserving relevant information for your specific use case.
      </p>
      
      <h3 className="text-2xl font-semibold text-gray-700">3. Split the data</h3>
      <p className="text-gray-600">
        Properly splitting your data into training, validation, and test sets is crucial for developing a robust and generalizable SLM. This division allows you to train your model, tune its hyperparameters, and evaluate its performance on unseen data, giving you a realistic assessment of how it will perform in real-world scenarios. For CompactLM, we've adopted a common split ratio of 70% for training, 15% for validation, and 15% for the test set.
      </p>
      <p className="text-gray-600">
        When splitting your data, it's important to ensure that each set is representative of the overall dataset. This means maintaining similar distributions of different topics, writing styles, or any other relevant characteristics across all three sets. Random splitting is often a good starting point, but for more complex datasets, you might need to use stratified sampling to ensure balanced representation of important categories or features.
      </p>
      <p className="text-gray-600">
        Consider the temporal aspects of your data, especially if it includes time-sensitive information. For instance, if you're working with news articles or social media posts, you might want to ensure that your test set consists of more recent data than your training set. This approach helps evaluate how well your model generalizes to new, unseen data and how it handles potential concept drift over time.
      </p>
      <p className="text-gray-600">
        For SLMs, which often work with limited data compared to larger models, it's particularly important to make efficient use of your dataset. Techniques like cross-validation can be valuable, especially if your dataset is small. K-fold cross-validation, where you divide your data into K subsets and train K different models (each using a different subset as the validation set), can provide a more robust evaluation of your model's performance and help in identifying potential overfitting issues.
      </p>
      
      <h3 className="text-2xl font-semibold text-gray-700">4. Implement data augmentation techniques</h3>
      <p className="text-gray-600">
        Data augmentation is a powerful technique for expanding your training dataset and improving your SLM's robustness and generalization capabilities. For CompactLM, we've implemented several augmentation techniques, including synonym replacement and back-translation. These methods help increase the diversity of our training data without requiring additional data collection.
      </p>
      <p className="text-gray-600">
        Synonym replacement involves randomly replacing words in the original text with their synonyms. This technique helps the model learn to understand different ways of expressing the same concept. When implementing synonym replacement, it's important to use a high-quality synonym dictionary or a pre-trained word embedding model to ensure that the replacements are semantically appropriate. You should also be careful not to replace domain-specific terms or named entities that carry important meaning.
      </p>
      <p className="text-gray-600">
        Back-translation is another effective augmentation technique, especially useful for improving the model's robustness to paraphrasing. This method involves translating the original text to another language and then back to the original language. The resulting text often contains slight variations in word choice and sentence structure while preserving the overall meaning. When implementing back-translation, consider using multiple intermediate languages to increase diversity. Also, be aware that this technique can introduce errors or change the meaning slightly, so it's important to manually review a sample of the augmented data.
      </p>
      <p className="text-gray-600">
        For SLMs, which often work with limited data, consider additional augmentation techniques that can help the model learn to handle variations it might encounter in real-world use. This could include adding random noise to the text (e.g., typos or grammatical errors), changing the order of sentences in longer texts, or generating new sentences using the existing vocabulary. Always ensure that your augmentation techniques are appropriate for your specific domain and use case, and validate that they're not introducing biases or errors that could negatively impact the model's performance.
      </p>
      
      {/* ... (continue with the rest of the component) ... */}
    </div>
  );
};

export default DataPreparation;