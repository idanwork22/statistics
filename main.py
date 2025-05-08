# -*- coding: utf-8 -*-

# קובץ Python לניתוח נתוני תאונות דרכים - כולל מגוון שיטות סטטיסטיות מתקדמות
# (רגרסיה לינארית, GLM, PCA, רגולריזציה, רגרסיה לוגיסטית, פואסונית, ניתוח שרידות, מבחנים א-פרמטריים, CART, Random Forest, GEE/LMM).
# התיעוד וההסברים ניתנים בהערות (בעברית) לאורך הקוד.

# Standard library imports
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Statsmodels imports
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable

# Scikit-learn imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# לשיפור נראות הפלט (לא חובה):
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 50)

# Configure matplotlib for non-interactive mode but still generate plots
plt.ioff()
plt.switch_backend('Agg')  # Use non-interactive backend
# Set a smaller figure size and reduce DPI for faster rendering
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100

# =============================================================================
# 1. טעינת הנתונים וסקירה ראשונית
# =============================================================================
# קריאת קובץ הנתונים (יש לוודא שהקובץ "crash_data.csv" זמין)
try:
    df = pd.read_csv('crash_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('crash_data.csv', encoding='latin1')
    except:
        df = pd.read_csv('crash_data.csv', encoding='iso-8859-8')

# Create Casualties column based on Injury Severity
severity_map = {
    'NO APPARENT INJURY': 0,
    'POSSIBLE INJURY': 1,
    'SUSPECTED MINOR INJURY': 1,
    'SUSPECTED SERIOUS INJURY': 2,
    'FATAL INJURY': 3
}
df['Casualties'] = df['Injury Severity'].map(severity_map).fillna(0)

# הצגת כמות רשומות ועמודות לדוגמה:
print("Number of records:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(3))

# =============================================================================
# 1.5 Exploratory Data Visualization
# =============================================================================
print("\n=== Creating exploratory visualizations of key variables ===")

# Create a visualization sample for exploratory plots
vis_explore_size = min(10000, len(df))
df_vis = df.sample(n=vis_explore_size, random_state=42) if len(df) > vis_explore_size else df.copy()

# 1. Distribution of Casualties
plt.figure(figsize=(8, 5))
casualty_counts = df_vis['Casualties'].value_counts().sort_index()
plt.bar(casualty_counts.index, casualty_counts.values)
plt.title('Distribution of Casualties')
plt.xlabel('Number of Casualties')
plt.ylabel('Count')
plt.xticks(range(int(df_vis['Casualties'].max())+1))
plt.savefig('casualties_distribution.png')
plt.close()
print("- Saved casualties distribution plot")

# 2. Top Weather Conditions Bar Chart
plt.figure(figsize=(10, 6))
weather_counts = df_vis['Weather'].value_counts().head(10)
weather_counts.plot(kind='barh')
plt.title('Top 10 Weather Conditions')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('top_weather_conditions.png')
plt.close()
print("- Saved top weather conditions plot")

# 3. Time-related visualizations if available
if 'Crash Date/Time' in df.columns:
    # Convert to datetime and extract hour
    try:
        df_vis['Crash_Hour'] = pd.to_datetime(df_vis['Crash Date/Time'], errors='coerce').dt.hour
        
        plt.figure(figsize=(10, 5))
        df_vis['Crash_Hour'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribution of Crashes by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Crashes')
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig('crashes_by_hour.png')
        plt.close()
        print("- Saved crashes by hour plot")
    except:
        print("- Could not create time-related plots due to datetime conversion issues")

# 4. Speed Limit vs Casualties boxplot
plt.figure(figsize=(10, 6))
df_vis_grouped = df_vis.groupby('Speed Limit')['Casualties'].mean().reset_index()
df_vis_grouped = df_vis_grouped[df_vis_grouped['Speed Limit'] > 0]  # Filter out zero speed limits
plt.bar(df_vis_grouped['Speed Limit'], df_vis_grouped['Casualties'])
plt.title('Average Casualties by Speed Limit')
plt.xlabel('Speed Limit')
plt.ylabel('Average Casualties')
plt.xticks(df_vis_grouped['Speed Limit'])
plt.savefig('speed_limit_casualties.png')
plt.close()
print("- Saved speed limit vs casualties plot")

# 5. Correlation heatmap of numerical variables
numeric_cols = df_vis.select_dtypes(include=[np.number]).columns
corr = df_vis[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()
print("- Saved correlation heatmap")

# 6. Missing data visualization
plt.figure(figsize=(12, 8))
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]
missing_percent = (missing_data / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing_data, 'Percent': missing_percent})
missing_df = missing_df.head(20)  # Top 20 columns with missing values

plt.barh(missing_df.index, missing_df['Percent'])
plt.title('Top 20 Columns with Missing Values (%)')
plt.xlabel('Percentage Missing')
plt.tight_layout()
plt.savefig('missing_data.png')
plt.close()
print("- Saved missing data visualization")

# 7. Surface Condition vs Casualties
if 'Surface Condition' in df_vis.columns:
    surface_casualties = df_vis.groupby('Surface Condition')['Casualties'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    surface_casualties.plot(kind='bar')
    plt.title('Average Casualties by Surface Condition')
    plt.xlabel('Surface Condition')
    plt.ylabel('Average Casualties')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('surface_casualties.png')
    plt.close()
    print("- Saved surface condition vs casualties plot")

# 8. Light Condition vs Casualties
if 'Light' in df_vis.columns:
    light_casualties = df_vis.groupby('Light')['Casualties'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    light_casualties.plot(kind='bar')
    plt.title('Average Casualties by Light Condition')
    plt.xlabel('Light Condition')
    plt.ylabel('Average Casualties')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('light_casualties.png')
    plt.close()
    print("- Saved light condition vs casualties plot")

print("Exploratory visualizations complete.\n")

# הסרת עמודות מזהה/תאריך אם אינן רלוונטיות (לדוגמה: מזהה ייחודי או תאריך שלא נשתמש בו במודל)
for col in ['Report Number', 'Local Case Number', 'Crash Date/Time']:
    if col in df.columns:
        df = df.drop(col, axis=1)

# =============================================================================
# 2. טיפול בערכים חסרים (Missing Data)
# =============================================================================
# נבדוק אם קיימים ערכים חסרים בכל עמודה:
missing_counts = df.isnull().sum()
print("\nMissing values per column:\n", missing_counts)

# טיפול בחסרים:
# תחילה, השלמה פשוטה: מספרי -> חציון; קטגוריאלי -> הערך השכיח.
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

for col in num_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
        
for col in cat_cols:
    if df[col].isnull().any():
        # Handle empty mode case safely
        mode_vals = df[col].mode()
        if not mode_vals.empty:
            df[col] = df[col].fillna(mode_vals[0])
        else:
            # If no mode exists, use a placeholder
            df[col] = df[col].fillna("Unknown")

print("\nMissing after simple imputation:", df.isnull().sum().sum(), "values remaining.")

# השלמה מרובה (Multiple Imputation) - ניישם IterativeImputer למשתנים מספריים כדוגמה.
if df.select_dtypes(include=[np.number]).isnull().any().any():
    imp = IterativeImputer(max_iter=10, random_state=0)
    df[num_cols] = imp.fit_transform(df[num_cols])
    print("Completed iterative imputation for numeric columns.")

# =============================================================================
# 3. דגימה (Sampling) למטרות ביצועים (אם נדרש)
# =============================================================================
# Using the full dataset for analysis
df_sample = df.copy()
print(f"\nAnalyzing full dataset with {len(df_sample)} records.\n")

# Create a visualization sample for plots (5000 records max for performance)
vis_sample_size = min(5000, len(df_sample))
vis_sample = df_sample.sample(n=vis_sample_size, random_state=42) if len(df_sample) > vis_sample_size else df_sample.copy()
print(f"Using a sample of {len(vis_sample)} records for visualization.\n")

# =============================================================================
# 4. רגרסיה לינארית מרובת משתנים (Linear Regression)
# =============================================================================
# נגדיר משתנה מטרה רציף, למשל מספר הנפגעים בתאונה (Casualties).
if 'Casualties' not in df_sample.columns:
    # אם לא קיים, ונמצאים נתוני הרוגים ופצועים, נחשב סכום
    if 'Fatalities' in df_sample.columns and 'Injuries' in df_sample.columns:
        df_sample['Casualties'] = df_sample['Fatalities'] + df_sample['Injuries']
# משתני חיזוי (מסבירים):
predictors = []
# משתנים מספריים:
if 'Vehicles' in df_sample.columns:
    predictors.append('Vehicles')
if 'Hour' in df_sample.columns:
    predictors.append('Hour')
# משתנים קטגוריים - נוסיף עם C() בפורמולת המודל ליצירת משתני דמה:
if 'Weather' in df_sample.columns:
    predictors.append('C(Weather)')
if 'Road_Type' in df_sample.columns:
    predictors.append('C(Road_Type)')
if 'Light_Condition' in df_sample.columns:
    predictors.append('C(Light_Condition)')
if 'Region' in df_sample.columns:
    predictors.append('C(Region)')
# בניית המודל ופתרונו:
formula = "Casualties ~ " + " + ".join(predictors)
model_lin = smf.ols(formula, data=df_sample)
results_lin = model_lin.fit()
print("Linear Regression Model Summary:")
print(results_lin.summary())
# מדדים חשובים:
print(f"R-squared: {results_lin.rsquared:.3f}, Adjusted R-squared: {results_lin.rsquared_adj:.3f}")
print(f"F-statistic p-value (overall model): {results_lin.f_pvalue:.4e}")
# דוגמה לפירוש מקדם:
if any('Vehicles' in term for term in predictors):
    coef = results_lin.params[[p for p in results_lin.params.index if 'Vehicles' in p][0]]
    print(f"Coefficient for 'Vehicles': {coef:.3f} (כלי רכב נוסף בתאונה משנה את מספר הנפגעים בכ-{coef:.3f} בממוצע)")
# בדיקות הנחות:
residuals = results_lin.resid
fitted = results_lin.fittedvalues

# 1. נורמליות השאריות - מבחן Jarque-Bera:
_, JB_p = stats.jarque_bera(residuals)
print(f"Jarque-Bera test p-value (residual normality): {JB_p:.4f}")
# 2. הומוסקדסטיות - מבחן Breusch-Pagan:
bp_test = sms.het_breuschpagan(residuals, results_lin.model.exog)
print(f"Breusch-Pagan test p-value (homoscedasticity): {bp_test[3]:.4f}")
# 3. רב-קוולינאריות - חישוב VIF:
X = pd.DataFrame(results_lin.model.exog, columns=results_lin.model.exog_names)
print("VIF values:")
for i in range(1, X.shape[1]):  # מדלגים על האינטרספט
    vif_val = variance_inflation_factor(X.values, i)
    print(f"{X.columns[i]}: {vif_val:.2f}")

# Create plot using visualization sample
vis_residuals = residuals.iloc[:vis_sample_size] if len(residuals) > vis_sample_size else residuals
vis_fitted = fitted.iloc[:vis_sample_size] if len(fitted) > vis_sample_size else fitted

# גרפים:
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.scatter(vis_fitted, vis_residuals, alpha=0.3, s=10)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")

plt.subplot(122)
sm.graphics.qqplot(vis_residuals, line='45', fit=True, ax=plt.gca())
plt.title("Normal Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig('residual_plots.png')
plt.close()
print("Diagnostic plots saved to 'residual_plots.png'")

# =============================================================================
# 5. בחירת משתנים (Feature Selection)
# =============================================================================
# שיטת Forward Selection לפי AIC:
available_terms = predictors.copy()
selected_terms = []
current_aic = smf.ols("Casualties ~ 1", data=df_sample).fit().aic
while available_terms:
    best_term = None
    best_aic = current_aic
    for term in available_terms:
        formula_try = "Casualties ~ " + " + ".join(selected_terms + [term])
        aic = smf.ols(formula_try, data=df_sample).fit().aic
        if aic < best_aic:
            best_aic = aic
            best_term = term
    if best_term:
        selected_terms.append(best_term)
        available_terms.remove(best_term)
        current_aic = best_aic
        print(f"Added {best_term}, AIC={current_aic:.2f}")
    else:
        break
print("Selected predictors (forward AIC):", selected_terms)
# ניתן להשתמש גם ב-backward elimination (הסרת המשתנה הכי פחות מובהק בכל צעד) או בשילוב (Stepwise).
# כמו כן, אפשר להשוות קריטריונים כמו BIC. בבחירת מודל סיווג, לעיתים משתמשים ב-ROC/AUC כמדד איכות.

# =============================================================================
# 6. רגולריזציה: Ridge, LASSO, Partial Least Squares
# =============================================================================
# נכין מטריצת X ו-y למודל רגרסיה (נמשיך עם חיזוי Casualties):
X = df_sample.drop(columns=['Casualties','Severity','Fatal_Accident'], errors='ignore')
X = pd.get_dummies(X, drop_first=True)
X = X.select_dtypes(include=[np.number])  # רק משתנים מספריים
y = df_sample['Casualties']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)
print("\nRidge coefficients:", dict(zip(X.columns, np.round(ridge.coef_,3))))
print("\nLasso coefficients:", dict(zip(X.columns, np.round(lasso.coef_,3))))
# ניתן לראות שב-Lasso חלק מהמקדמים מתאפסים (feature selection מובנה).
# PLS Regression:
pls = PLSRegression(n_components=2)
pls.fit(X_scaled, y)
print("\nPLS regression coefficients:", dict(zip(X.columns, np.round(pls.coef_.ravel(),3))))
# ה-PLS מפיק קומבינציות לינאריות של המשתנים המסבירים עם מקדמי רגרסיה, ככלי להתמודדות עם משתנים מתואמים.

# =============================================================================
# 7. ניתוח מרכיבים עיקריים (PCA) וניתוח גורמים (FA)
# =============================================================================
numeric_vars_for_pca = [col for col in ['Hour','Vehicles','Fatalities','Injuries','Casualties'] if col in df_sample.columns]
numeric_data = df_sample[numeric_vars_for_pca].fillna(0)

# Make sure we have enough data for PCA
try:
    n_components = min(3, len(df_sample) - 1, len(numeric_vars_for_pca))
    if n_components > 0:
        numeric_data_std = StandardScaler().fit_transform(numeric_data)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(numeric_data_std)
        print(f"\nPCA explained variance ratios: {np.round(pca.explained_variance_ratio_,3)}")
        loadings = pd.DataFrame(pca.components_.T, index=numeric_vars_for_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
        print("PCA loadings:\n", loadings)
        
        # סיבוב Varimax (FA):
        def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
            p, k = Phi.shape
            R = np.eye(k)
            for i in range(q):
                Lambda = Phi @ R
                u, s, vh = np.linalg.svd(Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
                R = u @ vh
                if s.sum() < tol:
                    break
            return Phi @ R
            
        rotated = pd.DataFrame(varimax(loadings.values), index=loadings.index, columns=loadings.columns)
        print("Rotated loadings (Varimax):\n", np.round(rotated,3))
    else:
        print("\nNot enough data for PCA. Skipping PCA analysis.")
except Exception as e:
    print(f"\nError in PCA: {e}")
    print("Skipping PCA analysis due to insufficient data.")

# =============================================================================
# 8. רגרסיה לוגיסטית (Logistic Regression)
# =============================================================================
# נגדיר משתנה יעד בינארי: האם התאונה קטלנית (Fatal_Accident).
try:
    if 'Fatal_Accident' not in df_sample.columns and 'Severity' in df_sample.columns:
        # בהנחה שחומרה מקודדת כך שהערך הגבוה מייצג קטלני (או אם יש עמודת Fatalities)
        if df_sample['Severity'].dtype != object:
            fatal_code = df_sample['Severity'].max()  # מניחים הקוד הגבוה ביותר = קטלנית
            df_sample['Fatal_Accident'] = (df_sample['Severity'] == fatal_code).astype(int)
            
    # Check if we have the required column
    if 'Fatal_Accident' not in df_sample.columns:
        print("\nNo 'Fatal_Accident' column available for logistic regression. Creating dummy column.")
        # Create dummy binary column based on Casualties > 0
        df_sample['Fatal_Accident'] = (df_sample['Casualties'] > 0).astype(int)
            
    # בניית מודל לוגיסטי:
    logit_predictors = [term for term in predictors if 'Casualties' not in term]  # מסירים את משתנה היעד מהרשימה אם מופיע
    
    # Make sure we have enough features for logistic regression
    if logit_predictors:
        formula_logit = "Fatal_Accident ~ " + " + ".join(logit_predictors)
        model_logit = smf.glm(formula_logit, data=df_sample, family=sm.families.Binomial())
        results_logit = model_logit.fit()
        print("\nLogistic Regression Model Summary:")
        print(results_logit.summary())
        
        # המקדמים מייצגים log-odds. נסב אותם ליחסי סיכויים (odds ratios):
        odds_ratios = np.exp(results_logit.params)
        print("Odds Ratios:\n", np.round(odds_ratios,3))
        
        # מבחני Wald: מוצגים בסיכום (p-value לכל מקדם).
        # מבחן יחס סבירות (Likelihood Ratio) למודל כולו:
        null_logit = smf.glm("Fatal_Accident ~ 1", data=df_sample, family=sm.families.Binomial()).fit()
        LR_stat = 2*(results_logit.llf - null_logit.llf)
        LR_p = 1 - stats.chi2.cdf(LR_stat, df=results_logit.df_model)
        print(f"Model Likelihood-Ratio test: Chi^2={LR_stat:.2f}, p-value={LR_p:.4e}")
        
        # Evaluate only if we have enough data
        if len(df_sample) > 10:
            # הערכת ביצועי המודל הלוגיסטי:
            y_true = df_sample['Fatal_Accident']
            y_pred = (results_logit.predict(df_sample) >= 0.5).astype(int)
            print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
            print("Classification Report:\n", classification_report(y_true, y_pred, digits=3))
            
            # ROC curve:
            y_score = results_logit.predict(df_sample)

            # Use a subset for visualization
            if len(y_true) > vis_sample_size:
                vis_indices = np.random.choice(len(y_true), vis_sample_size, replace=False)
                vis_y_true = y_true.iloc[vis_indices]
                vis_y_score = y_score.iloc[vis_indices]
            else:
                vis_y_true = y_true
                vis_y_score = y_score

            fpr, tpr, _ = roc_curve(vis_y_true, vis_y_score)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f'Logistic (AUC = {roc_auc:.2f})')
            plt.plot([0,1],[0,1],'--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Logistic Regression')
            plt.legend(loc="lower right")
            plt.savefig('logistic_roc.png')
            plt.close()
            print("ROC curve saved to 'logistic_roc.png'")
        else:
            print("Not enough data for ROC curve and confusion matrix analysis.")
    else:
        print("\nNo predictors available for logistic regression.")
except Exception as e:
    print(f"\nError in logistic regression: {e}")
    print("Skipping logistic regression analysis.")

# =============================================================================
# 9. רגרסיית פואסון (Poisson Regression)
# =============================================================================
try:
    # נשתמש במספר הנפגעים (Casualties) כמשתנה מטרה לספירה.
    if predictors:
        formula_pois = formula  # משתמשים באותה פורמולה כמו ברגרסיה הלינארית
        model_pois = smf.glm(formula_pois, data=df_sample, family=sm.families.Poisson())
        results_pois = model_pois.fit()
        print("\nPoisson Regression Model Summary:")
        print(results_pois.summary())
        # בדיקת Overdispersion:
        resid_pearson = results_pois.resid_pearson
        overdispersion = np.sum(resid_pearson**2) / results_pois.df_resid
        print(f"Overdispersion ratio = {overdispersion:.2f}")
    else:
        print("\nNo predictors available for Poisson regression.")
except Exception as e:
    print(f"\nError in Poisson regression: {e}")
    print("Skipping Poisson regression analysis.")

# =============================================================================
# 10. ניתוח שרידות (Survival Analysis)
# =============================================================================
# אין נתוני זמן עד אירוע או צנזורה ולכן נדלג על ניתוח שרידות, אך נזכיר:
print("\nSkipping Survival Analysis (no time-to-event data available)")

# =============================================================================
# 11. מבחנים א-פרמטריים (Non-parametric tests)
# =============================================================================
try:
    # Mann-Whitney U (Wilcoxon rank-sum) - הבדל בין שתי קבוצות בלתי תלויות:
    if 'Road_Type' in df_sample.columns:
        types = df_sample['Road_Type'].dropna().unique()
        if len(types) >= 2:
            g1 = df_sample[df_sample['Road_Type']==types[0]]['Casualties']
            g2 = df_sample[df_sample['Road_Type']==types[1]]['Casualties']
            if len(g1) > 0 and len(g2) > 0:
                mw_stat, mw_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                print(f"\nMann-Whitney U (Casualties by {types[0]} vs {types[1]} roads): U={mw_stat}, p-value={mw_p:.4f}")
                ks_stat, ks_p = stats.ks_2samp(g1, g2)
                print(f"Kolmogorov-Smirnov (Casualties dist {types[0]} vs {types[1]}): stat={ks_stat:.3f}, p-value={ks_p:.4f}")
            else:
                print(f"\nNot enough data for Mann-Whitney U test on Road_Type groups.")
        else:
            print(f"\nNot enough different Road_Type values for group comparison.")
    
    # Kruskal-Wallis H - הבדל בין מספר קבוצות:
    if 'Weather' in df_sample.columns:
        groups = [df_sample[df_sample['Weather']==w]['Casualties'] for w in df_sample['Weather'].dropna().unique()]
        valid_groups = [g for g in groups if len(g) > 0]
        if len(valid_groups) > 1:
            kw_stat, kw_p = stats.kruskal(*valid_groups)
            print(f"Kruskal-Wallis (Casualties by Weather): H={kw_stat:.3f}, p-value={kw_p:.4f}")
        else:
            print("Not enough weather groups with data for Kruskal-Wallis test.")
except Exception as e:
    print(f"\nError in non-parametric tests: {e}")
    print("Skipping non-parametric tests analysis.")

# =============================================================================
# 12. CART - עץ החלטה, ו-Random Forest
# =============================================================================
try:
    # מטרת סיווג: Fatal_Accident (כפי שהגדרנו לרגרסיה לוגיסטית).
    if 'Fatal_Accident' in df_sample.columns and X.shape[0] > 3:
        X_class = X.copy()
        y_class = df_sample['Fatal_Accident']
        
        # Check if we have enough samples of each class
        if len(y_class.unique()) > 1 and y_class.value_counts().min() > 0:
            # עץ החלטה:
            max_depth = min(4, X.shape[0] // 2)  # Limit tree depth based on sample size
            tree_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
            tree_clf.fit(X_class, y_class)
            print("\nDecision Tree feature importance:")
            print(dict(zip(X_class.columns, np.round(tree_clf.feature_importances_,3))))
            tree_rules = export_text(tree_clf, feature_names=list(X_class.columns))
            print("Tree structure:\n", tree_rules)
            
            # Random Forest (only if enough samples):
            if X.shape[0] >= 10:
                n_estimators = min(100, max(10, X.shape[0]))
                rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
                rf_clf.fit(X_class, y_class)
                rf_imp = dict(zip(X_class.columns, np.round(rf_clf.feature_importances_,3)))
                rf_imp = dict(sorted(rf_imp.items(), key=lambda item: item[1], reverse=True))
                print("Random Forest feature importance (sorted):")
                print(rf_imp)
                
                # הערכת המודלים (כאן על סט האימון עצמו לצורך הדגמה):
                print("\nDecision Tree Classification Report (training):\n", classification_report(y_class, tree_clf.predict(X_class), digits=3))
                print("Random Forest Classification Report (training):\n", classification_report(y_class, rf_clf.predict(X_class), digits=3))
                
                # ROC להשוואת Tree vs Forest:
                if hasattr(tree_clf, "predict_proba") and hasattr(rf_clf, "predict_proba"):
                    # Use visualization sample for plots
                    if len(y_class) > vis_sample_size:
                        vis_indices = np.random.choice(len(y_class), vis_sample_size, replace=False)
                        vis_X_class = X_class.iloc[vis_indices]
                        vis_y_class = y_class.iloc[vis_indices]
                        
                        # Get predictions for visualization sample
                        y_score_tree = tree_clf.predict_proba(vis_X_class)[:,1]
                        y_score_rf = rf_clf.predict_proba(vis_X_class)[:,1]
                        
                        fpr_tree, tpr_tree, _ = roc_curve(vis_y_class, y_score_tree)
                        fpr_rf, tpr_rf, _ = roc_curve(vis_y_class, y_score_rf)
                    else:
                        y_score_tree = tree_clf.predict_proba(X_class)[:,1]
                        y_score_rf = rf_clf.predict_proba(X_class)[:,1]
                        
                        fpr_tree, tpr_tree, _ = roc_curve(y_class, y_score_tree)
                        fpr_rf, tpr_rf, _ = roc_curve(y_class, y_score_rf)
                    
                    auc_tree = auc(fpr_tree, tpr_tree)
                    auc_rf = auc(fpr_rf, tpr_rf)
                    
                    plt.figure(figsize=(6, 5))
                    plt.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC={auc_tree:.2f})')
                    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.2f})')
                    plt.plot([0,1],[0,1],'--', color='gray')
                    plt.legend()
                    plt.title("ROC - Decision Tree vs Random Forest")
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.savefig('tree_vs_forest_roc.png')
                    plt.close()
                    print("ROC comparison saved to 'tree_vs_forest_roc.png'")
            else:
                print("Not enough samples for Random Forest model. Skipping Random Forest.")
        else:
            print("\nNot enough samples of each class for classification modeling.")
    else:
        print("\nInsufficient data for classification modeling.")
except Exception as e:
    print(f"\nError in CART/Random Forest: {e}")
    print("Skipping classification analysis.")

# =============================================================================
# 13. נתונים אורכיים / תלויי קבוצה (GEE, Mixed Models)
# =============================================================================
try:
    # אם קיימת תלות בין תצפיות (למשל תאונות מקובצות לפי אזור או זמן), נשתמש ב-GEE או LMM.
    if 'Region' in df_sample.columns:
        region_counts = df_sample['Region'].value_counts()
        
        # Check if we have enough samples in each region
        if len(region_counts) >= 2 and region_counts.min() >= 2:
            # GEE
            if 'Fatal_Accident' in df_sample.columns and logit_predictors:
                gee_model = GEE.from_formula(formula_logit, groups="Region", data=df_sample, family=sm.families.Binomial(), cov_struct=Exchangeable())
                gee_results = gee_model.fit()
                print("\nGEE Logistic (clusters=Region) summary:")
                print(gee_results.summary())
            
            # Mixed Model
            if predictors:
                mixed_formula = "Casualties ~ " + " + ".join([p for p in predictors if p != "C(Region)"])
                if mixed_formula != "Casualties ~ ":  # Check if we have predictors other than Region
                    mixed_model = smf.mixedlm(mixed_formula, data=df_sample, groups=df_sample['Region'])
                    mixed_results = mixed_model.fit(reml=False)
                    print("\nMixed Linear Model summary:")
                    print(mixed_results.summary())
                else:
                    print("\nNo predictors available for mixed model other than Region.")
        else:
            print("\nNot enough samples per Region for GEE/Mixed Models analysis.")
    else:
        print("\nNo Region column available for grouped data analysis.")
except Exception as e:
    print(f"\nError in GEE/Mixed Models: {e}")
    print("Skipping grouped data analysis.")

print("\nAnalysis complete!")
