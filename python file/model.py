# Bank Deposit Prediction Cleaned Pipeline
# All output labels for target/metrics MUST be binary (0/1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import optuna
from optuna.samplers import TPESampler

np.random.seed(42)

class DepositFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):  # penambahan fitur baru dengan perbedaan di notebook
        X = X.copy()
        X['is_senior'] = (X['age'] >= 65).astype(int)
        X['negative_balance'] = (X['balance'] < 0).astype(int)
        X['high_campaign'] = (X['campaign'] >= 10).astype(int)
        X['new_customer'] = (X['pdays'] == -1).astype(int)
        X['balance_per_age'] = X['balance'] / (X['age'] + 1)
        X['campaign_intensity'] = X['campaign'] / (X['pdays'].replace(-1, 1) + 1)
        X['prime_age'] = ((X['age'] >= 30) & (X['age'] <= 50)).astype(int)
        return X

class BankDepositPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.feature_generator = DepositFeatureGenerator()
        self.feature_names = None
    def setup_preprocessor(self):
        num_features = [
            'age', 'balance', 'campaign', 'pdays',
            'is_senior', 'negative_balance', 'high_campaign', 'new_customer',
            'balance_per_age', 'campaign_intensity', 'prime_age'
        ]
        cat_features = ['job', 'housing', 'loan', 'contact', 'month', 'poutcome']
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ], remainder="drop", verbose=False)

    def prepare_data(self, X_train, X_test, y_train, y_test, X_unseen=None):
        X_train_fe = self.feature_generator.fit_transform(X_train)
        X_test_fe = self.feature_generator.transform(X_test)
        if X_unseen is not None:
            X_unseen_fe = self.feature_generator.transform(X_unseen)
        self.setup_preprocessor()
        X_train_processed = self.preprocessor.fit_transform(X_train_fe)
        X_test_processed = self.preprocessor.transform(X_test_fe)
        self.feature_names = self.preprocessor.get_feature_names_out()
        if X_unseen is not None:
            X_unseen_processed = self.preprocessor.transform(X_unseen_fe)
            return X_train_processed, X_test_processed, X_unseen_processed
        return X_train_processed, X_test_processed
    def define_models(self):
        models_config = {
            'Logistic_Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'param_space': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'class_weight': [None, 'balanced']
                }
            },
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=42),
                'param_space': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': [None, 'balanced']
                }
            },
            'Decision_Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'param_space': {
                    'max_depth': [3, 5, 7, 10, 15],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'criterion': ['gini', 'entropy'],
                    'class_weight': [None, 'balanced']
                }
            },
            'Gradient_Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_space': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'param_space': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': [None, 'balanced']
                }
            }
        }
        return models_config
    def calculate_business_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        # Ensure all y_true and y_pred are binary (0/1)
        y_true_bin = pd.Series(y_true).map({1: 1, 'yes': 1, 0: 0, 'no': 0}).astype(int)
        y_pred_bin = pd.Series(y_pred).map({1: 1, 'yes': 1, 0: 0, 'no': 0}).astype(int)
        precision = precision_score(y_true_bin, y_pred_bin)
        recall = recall_score(y_true_bin, y_pred_bin)
        f1 = f1_score(y_true_bin, y_pred_bin)
        auc = roc_auc_score(y_true_bin, y_pred_proba)
        baseline_conversion = (y_true_bin == 1).mean()
        df_temp = pd.DataFrame({
            'actual': y_true_bin,
            'predicted_proba': y_pred_proba
        })
        df_temp = df_temp.sort_values('predicted_proba', ascending=False)
        top_decile = int(len(df_temp) * 0.1)
        top_decile_conversion = (df_temp.head(top_decile)['actual'] == 1).mean()
        lift = top_decile_conversion / baseline_conversion if baseline_conversion > 0 else 0
        cost_per_call = 10
        revenue_per_deposit = 1000
        tp = ((y_pred_bin == 1) & (y_true_bin == 1)).sum()
        fp = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()
        total_calls_with_model = tp + fp
        successful_deposits = tp
        wasted_calls = fp
        cost_saving_vs_random = len(y_true_bin) * 0.1 * cost_per_call
        revenue_uplift = successful_deposits * revenue_per_deposit
        net_benefit = revenue_uplift - (total_calls_with_model * cost_per_call)
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'baseline_conversion': baseline_conversion,
            'top_decile_lift': lift,
            'cost_saving': cost_saving_vs_random,
            'revenue_uplift': revenue_uplift,
            'net_benefit': net_benefit,
            'total_calls': total_calls_with_model,
            'successful_deposits': successful_deposits,
            'wasted_calls': wasted_calls
        }
    def train_models_with_optuna(self, X_train, X_test, y_train, y_test, n_trials=50):
        models_config = self.define_models()
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("=== Model Training dengan Optuna Optimization ===\n")
        # Make sure y_train is 1/0
        y_train_bin = pd.Series(y_train).map({1: 1, 'yes': 1, 0: 0, 'no': 0}).astype(int)
        y_test_bin = pd.Series(y_test).map({1: 1, 'yes': 1, 0: 0, 'no': 0}).astype(int)
        for name, config in models_config.items():
            print(f"Training {name}...")
            def objective(trial):
                params = {}
                for param, values in config['param_space'].items():
                    if any(v is None for v in values):
                        params[param] = trial.suggest_categorical(param, values)
                    elif isinstance(values[0], int):
                        params[param] = trial.suggest_int(param, min(values), max(values))
                    elif isinstance(values[0], float):
                        params[param] = trial.suggest_float(param, min(values), max(values))
                    else:
                        params[param] = trial.suggest_categorical(param, values)

                if config['model'].__class__.__name__ == 'SVC':   # kalau tidak distel seperti ini, akan error
                    # Buang 'probability' dari params jika ada, dan selalu set probability=True
                    params = {k: v for k, v in params.items() if k != 'probability'}
                    model = config['model'].__class__(random_state=42, probability=True, **params)
                else:
                    model = config['model'].__class__(random_state=42, **params)
                scores = cross_val_score(model, X_train, y_train_bin, cv=cv, 
                                         scoring='f1', n_jobs=-1)
                return scores.mean()
            study = optuna.create_study(direction='maximize', 
                                        sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            if config['model'].__class__.__name__ == 'SVC':
                best_model = config['model'].__class__(random_state=42, probability=True, **study.best_params)
            else:
                best_model = config['model'].__class__(random_state=42, **study.best_params)
            best_model.fit(X_train, y_train_bin)
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            metrics = self.calculate_business_metrics(y_test_bin, y_pred, y_pred_proba, name)
            results[name] = {
                'model': best_model,
                'best_params': study.best_params,
                'best_cv_score': study.best_value,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                **metrics
            }
            print(f"Best params: {study.best_params}")
            print(f"Best CV F1-Score: {study.best_value:.4f}")
            print(f"Test AUC: {metrics['auc_score']:.4f}")
            print(f"Test F1-Score: {metrics['f1_score']:.4f}")
            print(f"Top Decile Lift: {metrics['top_decile_lift']:.2f}x")
            print("-" * 60)
        self.models = results
        return results
    def select_best_model(self, primary_metric='f1_score', interpretability_weight=0.3):
        interpretability_scores = {
            'Logistic_Regression': 1.0,
            'Decision_Tree': 0.9,
            'Gradient_Boosting': 0.6,
            'Random_Forest': 0.5,
            'SVM': 0.3
        }
        model_scores = {}
        for name, results in self.models.items():
            performance_score = results[primary_metric]
            interpretability_score = interpretability_scores.get(name, 0.5)
            combined_score = ((1 - interpretability_weight) * performance_score + 
                              interpretability_weight * interpretability_score)
            model_scores[name] = {
                'combined_score': combined_score,
                'performance_score': performance_score,
                'interpretability_score': interpretability_score
            }
        sorted_models = sorted(model_scores.items(), 
                               key=lambda x: x[1]['combined_score'], reverse=True)
        print(f"\n=== Model Selection (Primary Metric: {primary_metric}) ===")
        print(f"{'Model':<20} {'Combined':<10} {'Performance':<12} {'Interpret.':<12}")
        print("-" * 60)
        for name, scores in sorted_models:
            print(f"{name:<20} {scores['combined_score']:<10.4f} "
                  f"{scores['performance_score']:<12.4f} {scores['interpretability_score']:<12.4f}")
        self.best_model_name = sorted_models[0][0]
        self.best_model = self.models[self.best_model_name]['model']
        print(f"\nSelected Best Model: {self.best_model_name}")
        return self.best_model_name, self.best_model
    def generate_comprehensive_report(self, y_test):
        best_results = self.models[self.best_model_name]
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE BUSINESS REPORT - {self.best_model_name}")
        print(f"{'='*60}")
        print(f"\n📊 MODEL PERFORMANCE METRICS:")
        print(f"   • Precision: {best_results['precision']:.3f} (minimizes wasted calls)")
        print(f"   • Recall: {best_results['recall']:.3f} (captures opportunities)")
        print(f"   • F1-Score: {best_results['f1_score']:.3f} (balanced performance)")
        print(f"   • AUC Score: {best_results['auc_score']:.3f} (overall discrimination)")
        print(f"\n💰 BUSINESS IMPACT SIMULATION:")
        print(f"   • Top Decile Lift: {best_results['top_decile_lift']:.1f}x baseline")
        print(f"   • Total Calls Needed: {best_results['total_calls']:,}")
        print(f"   • Successful Deposits: {best_results['successful_deposits']:,}")
        print(f"   • Wasted Calls: {best_results['wasted_calls']:,}")
        print(f"   • Estimated Revenue Uplift: ${best_results['revenue_uplift']:,}")
        print(f"   • Estimated Net Benefit: ${best_results['net_benefit']:,}")
        y_pred_proba = best_results['y_pred_proba']
        print(f"\n👥 CUSTOMER SEGMENTATION:")
        high_prob = (y_pred_proba >= 0.7).sum()
        medium_prob = ((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)).sum()
        low_prob = (y_pred_proba < 0.3).sum()
        total = len(y_pred_proba)
        print(f"   • High Potential (≥70%): {high_prob:,} customers ({high_prob/total:.1%})")
        print(f"   • Medium Potential (30-70%): {medium_prob:,} customers ({medium_prob/total:.1%})")
        print(f"   • Low Potential (<30%): {low_prob:,} customers ({low_prob/total:.1%})")
        print(f"\n🎯 ACTIONABLE RECOMMENDATIONS:")
        print(f"   1. Focus telemarketing on HIGH potential segment")
        print(f"   2. Use digital channels for MEDIUM potential segment")
        print(f"   3. Avoid LOW potential segment to save costs")
        print(f"   4. Expected campaign efficiency improvement: {best_results['top_decile_lift']:.1f}x")
        return best_results
    def plot_comprehensive_analysis(self, y_test):
        best_results = self.models[self.best_model_name]
        y_pred_proba = best_results['y_pred_proba']
        y_pred = best_results['y_pred']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive Model Analysis - {self.best_model_name}', 
                    fontsize=16, fontweight='bold')
        model_names = list(self.models.keys())
        f1_scores = [self.models[name]['f1_score'] for name in model_names]
        bars = axes[0,0].bar(model_names, f1_scores)
        axes[0,0].set_title('Model Performance Comparison (F1-Score)')
        axes[0,0].set_ylabel('F1-Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        best_idx = model_names.index(self.best_model_name)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(0.8)

        # 2. ROC Curve
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        y_test_bin = pd.Series(y_test).map({1: 1, 'yes': 1, 0: 0, 'no': 0}).astype(int)
        fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba)
        axes[0,1].plot(fpr, tpr, linewidth=2, label=f'AUC = {best_results["auc_score"]:.3f}')
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin, y_pred_proba)
        axes[0,2].plot(recall_curve, precision_curve, linewidth=2)
        axes[0,2].set_xlabel('Recall')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].set_title('Precision-Recall Curve')
        axes[0,2].grid(True, alpha=0.3)

        # 4. Confusion Matrix
        cm = confusion_matrix(y_test_bin, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title('Confusion Matrix')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        axes[1,0].set_xticklabels(['No', 'Yes'])
        axes[1,0].set_yticklabels(['No', 'Yes'])

        # 5. Probability Distribution
        axes[1,1].hist(y_pred_proba[y_test_bin==0], bins=30, alpha=0.7, 
                      label='No Deposit', density=True, color='red')
        axes[1,1].hist(y_pred_proba[y_test_bin==1], bins=30, alpha=0.7, 
                      label='Deposit', density=True, color='green')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Probability Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # 6. Customer Segmentation Pie
        segments = ['High\n(≥70%)', 'Medium\n(30-70%)', 'Low\n(<30%)']
        segment_counts = [
            (y_pred_proba >= 0.7).sum(),
            ((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)).sum(),
            (y_pred_proba < 0.3).sum()
        ]
        colors = ['green', 'orange', 'red']
        axes[1,2].pie(segment_counts, labels=segments, colors=colors, autopct='%1.1f%%')
        axes[1,2].set_title('Customer Segmentation')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        self.plot_business_metrics()


    def plot_business_metrics(self):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            model_names = list(self.models.keys())
            # Business metrics comparison
            metrics = ['precision', 'recall', 'f1_score', 'top_decile_lift']
            metric_values = {metric: [self.models[name][metric] for name in model_names] for metric in metrics}
            x = np.arange(len(model_names))
            width = 0.2
            for i, metric in enumerate(metrics):
                offset = (i - len(metrics)/2) * width
                axes[0].bar(x + offset, metric_values[metric], width, label=metric.replace('_', ' ').title())
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Business Metrics Comparison')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(model_names, rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            # ROI comparison (estimated)
            roi_values = [self.models[name]['net_benefit'] for name in model_names]
            bars = axes[1].bar(model_names, roi_values)
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Estimated Net Benefit ($)')
            axes[1].set_title('Estimated Business Impact')
            axes[1].tick_params(axis='x', rotation=45)
            best_idx = model_names.index(self.best_model_name)
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.8)
            plt.tight_layout()
            plt.show()

def main():
    """Function utama untuk menjalankan complete pipeline"""

    print("🏦 BANK DEPOSIT PREDICTION PIPELINE")
    print("=" * 50)

    # LANGKAH 1: Load data
    print("📁 Loading data_bank.csv...")

    df = pd.read_csv('data_bank.csv')

    df['deposit'] = df['deposit'].map({'no': 0, 'yes': 1})

    # LANGKAH 2: Eksplorasi data awal
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Target distribution:\n{df['deposit'].value_counts()}")

    # LANGKAH 3: Split data
    print("\n🔄 Splitting data...")
    # Split untuk data unseen (20%) dan data seen (80%)
    data_seen, data_unseen = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['deposit']
    )
    # Split data seen menjadi train (64%) dan test (16%)
    data_train, data_test = train_test_split(
        data_seen, test_size=0.2, random_state=42, stratify=data_seen['deposit']
    )

    # Pisahkan features dan target
    X_train = data_train.drop(columns=["deposit"])
    y_train = data_train["deposit"]
    X_test = data_test.drop(columns=["deposit"])
    y_test = data_test["deposit"]
    X_unseen = data_unseen.drop(columns=["deposit"])
    y_unseen = data_unseen["deposit"]

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Unseen set: {X_unseen.shape[0]} samples")

    # LANGKAH 4: Initialize predictor
    predictor = BankDepositPredictor()

    # LANGKAH 5: Prepare data dengan feature engineering
    print("\n⚙️ Preparing data dengan feature engineering...")
    X_train_processed, X_test_processed, X_unseen_processed = predictor.prepare_data(
        X_train, X_test, y_train, y_test, X_unseen
    )

    print(f"Processed features: {X_train_processed.shape[1]}")

    # LANGKAH 6: Train models dengan Optuna
    print("\n🤖 Training models dengan hyperparameter optimization...")
    results = predictor.train_models_with_optuna(
        X_train_processed, X_test_processed, y_train, y_test, n_trials=30
    )

    # LANGKAH 7: Select best model
    print("\n🏆 Selecting best model...")
    best_model_name, best_model = predictor.select_best_model(primary_metric='f1_score')

    # LANGKAH 8: Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    report = predictor.generate_comprehensive_report(y_test)

    # LANGKAH 9: Plot analysis
    print("\n📊 Generating visualizations...")
    predictor.plot_comprehensive_analysis(y_test)

    # LANGKAH 10: Test pada unseen data
    print("\n🔮 Testing on unseen data...")
    y_unseen_pred = best_model.predict(X_unseen_processed)
    y_unseen_proba = best_model.predict_proba(X_unseen_processed)[:, 1]

    # Metrics pada unseen data
    unseen_metrics = predictor.calculate_business_metrics(
        y_unseen, y_unseen_pred, y_unseen_proba, best_model_name
    )

    print(f"\n🎯 UNSEEN DATA PERFORMANCE:")
    print(f"   • F1-Score: {unseen_metrics['f1_score']:.3f}")
    print(f"   • AUC Score: {unseen_metrics['auc_score']:.3f}")
    print(f"   • Top Decile Lift: {unseen_metrics['top_decile_lift']:.1f}x")

    # LANGKAH 11: Simpan hasil prediksi
    print("\n💾 Saving predictions...")

    # Buat dataframe hasil
    results_df = pd.DataFrame({
        'customer_id': range(len(y_unseen)),
        'actual': y_unseen.values,
        'predicted': y_unseen_pred,
        'probability': y_unseen_proba,
        'segment': pd.cut(
            y_unseen_proba, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High']
        )
    })

    # Save ke CSV
    results_df.to_csv('deposit_predictions_unseen.csv', index=False)
    print("Predictions saved to deposit_predictions_unseen.csv")

if __name__ == "__main__":
    main()
