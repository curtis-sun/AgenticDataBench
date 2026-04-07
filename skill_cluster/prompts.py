SUMMARIZE_QA_SYS_PROMPT = '''Given a Stack Overflow question and its answer on data science, break down the solution into a series of clear, actionable steps. Each step should be a concise, natural language phrase that describes a specific action or detail. **Do not include any raw code blocks**—instead, describe code actions in natural language, specifying parameters and functions as needed. Include all relevant code specifics, such as function parameters (e.g., test_size=0.2 in train_test_split(X, y, test_size=0.2)), to ensure the steps are complete and implementable. The steps should be self-contained, logically ordered, and collectively cover every aspect of the solution process. 

For example:

### Step-by-Step Solution:

1. **Understand the Problem**: Recognize that predicting the sum of two numbers is a regression task, not a classification task.

2. **Choose the Right Algorithm**: Replace the Decision Tree Classifier with a Linear Regression model, which is suitable for predicting continuous outcomes.

3. **Import Necessary Libraries**: Ensure you have imported `LinearRegression` from `sklearn.linear_model` and the appropriate metrics such as `mean_squared_error` and `r2_score`.

4. **Prepare the Data**: Separate the features (columns A and B) into `X` and the target (column C) into `y`. The data is already clean, so no further preprocessing is needed.

5. **Split the Data**: Use `train_test_split` with a test size of 0.2 to divide the dataset into training and test sets.

6. **Train the Model**: Fit the Linear Regression model on the training data.

7. **Make Predictions**: Use the trained model to predict the target values for the test set.

8. **Evaluate the Model**: Calculate the performance using metrics appropriate for regression. Compute the Mean Squared Error (MSE) and the R² score. The R² score should be close to 1, indicating an excellent fit.

Avoid including any extra explanations.
'''

STACKOVERFLOW_QA_PROMPT = '''### Question: {title}

{question_body}

### Answer:

{answer_body}'''

QA_SYS_PROMPT = '''Given a data science task and its complete Python implementation, break down the solution into a series of clear, actionable steps. Each step should be a concise, natural language phrase that describes a specific action or detail. **Do not include any raw code blocks**—instead, describe code actions in natural language, specifying parameters and functions as needed. Include all relevant code specifics, such as function parameters (e.g., test_size=0.2 in train_test_split(X, y, test_size=0.2)), to ensure the steps are complete and implementable. The steps should be self-contained, logically ordered, and collectively cover every aspect of the solution process. 

For example:

### Step-by-Step Solution:

1. **Understand the Problem**: Recognize that predicting the sum of two numbers is a regression task, not a classification task.

2. **Choose the Right Algorithm**: Replace the Decision Tree Classifier with a Linear Regression model, which is suitable for predicting continuous outcomes.

3. **Import Necessary Libraries**: Ensure you have imported `LinearRegression` from `sklearn.linear_model` and the appropriate metrics such as `mean_squared_error` and `r2_score`.

4. **Prepare the Data**: Separate the features (columns A and B) into `X` and the target (column C) into `y`. The data is already clean, so no further preprocessing is needed.

5. **Split the Data**: Use `train_test_split` with a test size of 0.2 to divide the dataset into training and test sets.

6. **Train the Model**: Fit the Linear Regression model on the training data.

7. **Make Predictions**: Use the trained model to predict the target values for the test set.

8. **Evaluate the Model**: Calculate the performance using metrics appropriate for regression. Compute the Mean Squared Error (MSE) and the R² score. The R² score should be close to 1, indicating an excellent fit.

Avoid including any extra explanations.
'''

QA_PROMPT = '''### Task:
{question}

### Implementation:
```python
{answer}
```'''

EXTRACT_SYS_PROMPT = '''You will be provided with some solution steps for data science tasks. Your task is to extract and summarize all essential skills required, ensuring each skill is a concise and specific phrase. Carefully examine each solution step to comprehensively identify all necessary capabilities. Each skill should represent a capability relevant to a multiple tasks, avoiding overly general terms. Focus on the core technical or methodological action. Output a python list, each skill as a string, encapsulated with ```python and ```.

For example:

```python
[
    "Incremental training with partial_fit",
    "Negative sampling for large output spaces",
    "Random initialization of embeddings",
    "Training embedding layers via backpropagation",
    "Understanding gradient computation in TensorFlow",
    "Domain adaptation in transfer learning",
    "Dynamic model updating with new embeddings",
    "Setting batch sizes in data pipelines and model training",
    "Understanding convolution operations in CNNs",
    "Understanding batch size and epoch relationships"
]
```'''

EXTRACT_PROMPT = '''{steps}'''

CLUSTER_SYS_PROMPT = '''You will be provided with some solution steps for data science tasks, and a list of relevant skills. Your task is to group the steps into clusters based on the skills they involve. Each cluster should correspond to a specific skill from the provided list, and should contain the indexes of the steps that primarily use that skill.

Output as a Python dictionary encapsulated with ```python and ```, where each key is a skill (string) and each value is a list of the step indexes assigned to that cluster.
'''

CLUSTER_PROMPT = '''
Steps:
{steps}

Skills:
{skills}'''

CLUSTER_UNSUPERVISED_SYS_PROMPT = '''You will be provided with some data science entries, each in the format:
Skill <i>:\n**<skill name>**: <relevant solution step>
The text inside `**...**` is the **skill**. The text after the second colon provides a **solution step** that explains how the skill is applied.

Your task is to cluster these skills (i.e., the bolded phrases) based on their underlying technical or methodological actions. A single skill may belong to multiple clusters if it reflects multiple distinct technical or methodological actions. Ensure every skill is assigned to **at least one** cluster. Do not omit any skill.

Output as a Python dictionary encapsulated with ```python and ```, where each key is a cluster name (string) and each value is a list of the step indexes assigned to that cluster.'''

CLUSTER_UNSUPERVISED_PROMPT = '''{steps}'''

DESCRIBE_SKILL_SYS_PROMPT = '''You are given a data-science skill and several few-shot examples illustrating how the skill is applied.
Write a single, concise paragraph that defines the skill.
Generalize beyond the examples: use them only to infer the underlying capability, not to restate them.
Avoid mentioning the examples explicitly.
Be clear, precise, and domain-appropriate.'''

DESCRIBE_SKILL_USER_PROMPT = '''
## Skill: {skill}

{examples}'''