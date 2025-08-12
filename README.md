# Deposit Campaign Prediction

## 1. Business Understanding

### 1.1 Context
The bank faces challenges in increasing sales of new deposit products. Marketing resources such as telemarketing and promotions are limited, so a precise targeting strategy is needed to improve both effectiveness and efficiency of the campaign budget.

### 1.2 Problem Statement
Marketing campaigns are often conducted for all customers without prioritization. This leads to many ineffective calls and promotions because customers are either uninterested or not suitable for the product.

**Key Questions:**
1. Which customers are most likely to open a new deposit account if contacted?
2. Does the prediction truly reflect the customer’s actual interest?

### 1.3 Goals
- Build a predictive model to estimate the conversion probability for each customer.
- Increase conversion rate while reducing marketing costs.
- Prioritize telemarketing efforts towards **high potential leads**.

### 1.4 Analytical Approach
- **Identify Key Factors:** Analyze data to determine factors influencing deposit opening.
- **Detect Potential Customer Patterns:** Compare behaviors of customers who opened a deposit vs. those who did not.
- **Create Additional Indicators:** Add metrics like contact intensity and time since last contact.
- **Segment by Conversion Potential:** Group customers based on likelihood to convert.
- **Focus on High-Potential Segments:** Ensure high-probability leads are not missed.
- **Adapt to Behavioral Changes:** Continuously adjust strategy as customer behavior evolves.

### 1.5 Main Evaluation Metrics
- **Precision** → Minimize wasted calls.
- **Recall** → Avoid missing valuable opportunities.
- **F1-score** → Balance cost efficiency and revenue uplift.
- **Lift (Top Decile Lift)** → Evaluate the model's ability to identify the most promising leads.
- **Business Impact Simulation** → Translate predictions into tangible business outcomes.

**Example Business Impact Formulas:**
- **Cost Saving:** `(Number of saved calls) × (Cost per call)`
- **Revenue Uplift:** `(Additional deposits) × (Average deposit value)`
- **ROI:** `(Revenue uplift − Model implementation cost) / Model implementation cost`

### 1.6 Expected Outcomes
- Increase new deposit sales without increasing marketing expenses.
- Campaign efficiency: fewer calls, more deals.
- Marketing dashboard with prioritized lead scores.

### 1.7 Limitations & Challenges
- Balanced target distribution but limited features.
- No daily transaction behavior data available.
- Batch campaign data only (not real-time).

---

## 2. Data Understanding

**Dataset:** `data_bank.csv`  
- **Rows:** 7,813  
- **Columns:** 11  
- **Target:** `deposit` (yes/no)  
- **Numeric features:** `age`, `balance`, `campaign`, `pdays`  
- **Categorical features:** `job`, `housing`, `loan`, `contact`, `month`, `poutcome`

**Key Statistics:**
- Average age: 41 years (min: 18, max: 95)
- Average balance: 1,512 (min: -6,847, max: 66,653)
- Average contacts: 2–3 per customer, max 63
- `pdays`: Many entries at -1 (never contacted before)

---

## 3. Data Preparation

### 3.1 Data Splitting
- **Unseen data:** 20% (final evaluation only, untouched during training)
- **Train/test split:** Remaining 80% split into 80:20

### 3.2 Feature Engineering
Additional features created:
- `is_senior` → Age ≥ 65
- `negative_balance` → Balance < 0
- `high_campaign` → Contacts ≥ 10
- `new_customer` → `pdays` = -1

### 3.3 Preprocessing
- **Numeric:** Median imputation → Standard scaling
- **Categorical:** Fill missing with `'unknown'` → One-Hot Encoding (drop first)

---

## 4. Modeling
Machine learning classification models were applied (e.g., Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning via Optuna.

**Why Logistic Regression was selected as final model:**
- High interpretability: easy to explain feature influence on prediction.
- Well-suited for binary classification with probability-based targeting.

---

## 5. Evaluation
Main evaluation metrics:
- **Precision**
- **Recall**
- **F1-score**
- **Lift (Top Decile Lift)**

---

## 6. Business Impact
- **Cost Saving:** Reduce number of ineffective calls.
- **Revenue Uplift:** Increase number of new deposits.
- **ROI:** Measure net financial benefit of model deployment.

---

## 7. How to Run
```bash
# Clone repository
git clone https://github.com/<username>/<repo_name>.git
cd <repo_name>

# Install dependencies
pip install -r requirements.txt

# Run model training
python src/model.py

