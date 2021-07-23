# Terms_Conditions_Summarisation

The code accompanying my UCL MSc Business Analytics Dissertation/ Consulting Project with Amplifi.

We  propose a framework which can effectively summarise terms and conditions, condensing the contracts to 12% of their original length while also reducing the complexity of the language. 

The framework consists of a two-step approach to summarisation. First, we classify whether individual clauses in the terms and conditions are important or not. Second, we create summaries of the important clauses.

Using relevant domain data in addition to self-annotated data, we compare traditional machine learning techniques and Transformers for the classification task. We further explore both extractive and abstractive methods for text summarisation. 

We compare the results for both methods against various baselines. The results suggest that fine-tuning a transformer model for abstractive summarisation produces the best summaries of the terms and conditions. We conclude that more task-specific data would further increase the performance and generalisability of the model.
