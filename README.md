# Governed Prompt Gateway

## Local banking chatbot foundation

Dataset preparation, FAQ retrieval, and a Streamlit chat interface are implemented.
They run locally with no model, API key, or cloud account. The source Kaggle CSV
is included, along with 144 cleaned question/answer records and quality reports.

From the project folder, using Python 3.13:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m banking_chatbot prepare
python -m banking_chatbot search "How do I activate my debit card?"
python -m banking_chatbot evaluate
python -m unittest discover -s tests -v
```

Launch the chat interface:

```bash
python -m streamlit run app.py
```

Response caching defaults to exact matches. Semantic reuse uses a deterministic
local embedding (no download or external embedding API), always checks exact
matches first, and only compares entries with the same policy, sources, model,
prompt, knowledge-base, conversation, tenant, and embedding versions. Configure
the staged rollout before launch:

```bash
export CACHE_MODE=semantic-shadow  # off | exact | semantic-shadow | semantic
export SEMANTIC_CACHE_THRESHOLD=0.90
export SEMANTIC_CACHE_MARGIN=0.05
python -m streamlit run app.py
```

Use `semantic-shadow` to measure candidate quality without serving semantic
hits. Use `semantic` only after calibrating the threshold for the deployment's
queries; ambiguous candidates within the configured margin are rejected.

Open http://127.0.0.1:8501. The chat shows retrieved question/answer pairs, similarity
scores, expandable source details, session history, and a reset button. It uses no
LLM; each question is searched independently.

The project already has a local `.venv` with dependencies installed in this checkout.
Activate it to skip installation. After installation, all commands work offline.

- [Implementation and usage](docs/banking-chatbot.md)
- [Dataset provenance and license](data/banking/README.md)
- [Generated quality report](reports/banking_data_quality.md)
- [Manual source review](reports/banking_source_review.md)
- [Retrieval development results](reports/retrieval_dev.json)

Search returns reference records and similarity scores, not an LLM-generated answer.
This is synthetic educational data, not the policies or contact details of a real bank.

---

## Original project README template

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Taylor Nguyen    | @taylornguyen | Data exploration, visualization, overall project coordination            |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
