from fpdf import FPDF
import json

GITHUB_URL = "https://github.com/piyali-sarkar/wine-classifier"
STREAMLIT_URL = "http://localhost:8501"  # local deployment

# images (verified order)
IMG_APP_DECISION_TREE = "docs/image1.png"   # local app - Decision Tree
IMG_APP_NAIVE_BAYES   = "docs/image2.png"   # local app - Naive Bayes
IMG_BLOCKED_BITSLAB   = "docs/image3.png"   # Streamlit blocked on BITS Lab (Firefox)
IMG_BLOCKED_CHROME    = "docs/image4.png"   # Streamlit blocked on Chrome (BITS email)


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "ML Assignment 2 - Wine Quality Classifier", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def section(self, title):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def link_line(self, label, url):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, f"{label}: ", new_x="END")
        self.set_font("Helvetica", "U", 10)
        self.set_text_color(0, 0, 200)
        self.cell(0, 7, url, link=url, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [self.epw / len(headers)] * len(headers)
        # header row
        self.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C")
        self.ln()
        # data rows
        self.set_font("Helvetica", "", 8)
        for row in rows:
            max_lines = 1
            for i, cell in enumerate(row):
                lines = self.multi_cell(col_widths[i], 5, str(cell), border=0, dry_run=True, output="LINES")
                max_lines = max(max_lines, len(lines))
            row_h = max_lines * 5
            x_start = self.get_x()
            y_start = self.get_y()
            for i, cell in enumerate(row):
                self.set_xy(x_start + sum(col_widths[:i]), y_start)
                self.multi_cell(col_widths[i], row_h / max_lines, str(cell), border=1)
            self.set_xy(x_start, y_start + row_h)


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# --- 1. Links ---
pdf.section("1. Submission Links")
pdf.link_line("GitHub Repository", GITHUB_URL)
pdf.link_line("Streamlit App (Local)", STREAMLIT_URL)

# --- 2. Deployment Note ---
pdf.section("2. Note on Streamlit Cloud Deployment")
pdf.body_text(
    "My Streamlit Community Cloud account has been locked/blocked. When I try to sign in "
    "(both via my BITS email 2025AA05704@wilp.bits-pilani.ac.in and on BITS Virtual Lab), "
    "I get the error: 'Access blocked, please contact support.' "
    "I have contacted Streamlit support but the issue has not been resolved before the deadline. "
    "The BITS Virtual Lab was also not working for deployment.\n\n"
    "As a result, I ran the Streamlit app locally (localhost:8501) and have attached screenshots "
    "as proof that the app is fully functional."
)

pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 7, "Streamlit Cloud - Access Blocked (Chrome, BITS email):", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.image(IMG_BLOCKED_CHROME, w=150)
pdf.ln(4)

pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 7, "Streamlit Cloud - Access Blocked (BITS Virtual Lab, Firefox):", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.image(IMG_BLOCKED_BITSLAB, w=170)
pdf.ln(4)

# --- 3. App Screenshots (Local) ---
pdf.section("3. Streamlit App Screenshots (Local Deployment)")

pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 7, "App running locally - Decision Tree model:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.image(IMG_APP_DECISION_TREE, w=170)
pdf.ln(4)

pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 7, "App running locally - Naive Bayes model:", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.image(IMG_APP_NAIVE_BAYES, w=170)
pdf.ln(4)

# --- 4. README Content ---
pdf.section("4. README Content")

# Problem Statement
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "Problem Statement", new_x="LMARGIN", new_y="NEXT")
pdf.body_text(
    'Predict whether a wine is "good" (quality score >= 7) or "not good" (quality < 7) '
    "based on its physicochemical properties. This is a binary classification task using "
    "6 different ML models."
)

# Dataset
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "Dataset", new_x="LMARGIN", new_y="NEXT")
pdf.body_text(
    "Source: UCI Wine Quality Dataset\n"
    "Samples: 6497 (1599 red + 4898 white)\n"
    "Features: 12 (11 physicochemical properties + wine type)\n"
    "Target: Binary - 1 if quality >= 7 (good), 0 otherwise\n"
    "Class split: 1277 good, 5220 not good (imbalanced)"
)

feat_headers = ["#", "Feature", "Description"]
feat_rows = [
    ["1", "fixed acidity", "tartaric acid (g/dm3)"],
    ["2", "volatile acidity", "acetic acid (g/dm3)"],
    ["3", "citric acid", "(g/dm3)"],
    ["4", "residual sugar", "(g/dm3)"],
    ["5", "chlorides", "sodium chloride (g/dm3)"],
    ["6", "free sulfur dioxide", "(mg/dm3)"],
    ["7", "total sulfur dioxide", "(mg/dm3)"],
    ["8", "density", "(g/cm3)"],
    ["9", "pH", "acidity level"],
    ["10", "sulphates", "potassium sulphate (g/dm3)"],
    ["11", "alcohol", "(% vol)"],
    ["12", "type", "0 = red, 1 = white"],
]
pdf.add_table(feat_headers, feat_rows, col_widths=[10, 45, 50])
pdf.ln(6)

# Model Comparison
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "Model Comparison", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

metrics = json.load(open("model/metrics.json"))
comp_headers = ["ML Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
comp_rows = []
name_map = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "KNN": "KNN",
    "Naive Bayes": "Naive Bayes",
    "Random Forest": "Random Forest (Ensemble)",
    "XGBoost": "XGBoost (Ensemble)",
}
for key, display in name_map.items():
    m = metrics[key]
    comp_rows.append([display, m["Accuracy"], m["AUC"], m["Precision"], m["Recall"], m["F1"], m["MCC"]])

comp_widths = [42, 20, 18, 22, 18, 18, 18]
pdf.add_table(comp_headers, comp_rows, col_widths=comp_widths)
pdf.ln(6)

# Observations
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "Observations", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

observations = {
    "Logistic Regression": "High accuracy but very low recall (0.26) - barely identifies good wines. Linear boundary struggles with imbalanced data. AUC (0.80) shows decent ranking but default threshold is poor for minority class.",
    "Decision Tree": "Balanced precision-recall (0.63/0.64) with decent F1. Tends to overfit, shown by lower AUC (0.77) vs ensembles. Captures non-linear patterns logistic regression misses.",
    "KNN": "Moderate performance overall. AUC (0.83) is better than Decision Tree for ranking, but hard classification at k=5 loses some quality. Sensitive to feature scaling.",
    "Naive Bayes": "Lowest accuracy (0.73) due to many false positives. Independence assumption fails since wine features are correlated. But has second-highest recall (0.62), catching more good wines.",
    "Random Forest (Ensemble)": "Best overall - highest accuracy (0.89), AUC (0.91), precision (0.83), MCC (0.63). Bagging reduces overfitting. Conservative predictions keep false positives low.",
    "XGBoost (Ensemble)": "Second best, close to Random Forest. Slightly lower precision but similar recall and AUC (0.90). Boosting helps with hard samples. MCC (0.60) confirms solid performance.",
}

obs_headers = ["ML Model", "Observation"]
obs_rows = [[k, v] for k, v in observations.items()]
obs_widths = [42, 148]
pdf.add_table(obs_headers, obs_rows, col_widths=obs_widths)
pdf.ln(6)

# How to Run
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "How to Run", new_x="LMARGIN", new_y="NEXT")
pdf.body_text(
    "pip install -r requirements.txt\n"
    "python train.py\n"
    "streamlit run app.py"
)

out_path = "docs/ML_Assignment_2_Submission.pdf"
pdf.output(out_path)
print(f"PDF saved to {out_path}")
