"""Language Detection System: character n-gram TF-IDF + classical ML classifiers.

Usage:
  python src/detector.py train            # train, evaluate, save model + plots
  python src/detector.py predict "text"   # detect language of a sentence
"""
import sys, os, json, time, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "dataset.csv")
MODEL = os.path.join(ROOT, "model.joblib")
RES = os.path.join(ROOT, "results")

def make_pipeline(clf):
    # char n-grams of 1-3 inside word boundaries: language-agnostic, no tokenizer needed
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), lowercase=True, sublinear_tf=True)
    return Pipeline([("tfidf", vec), ("clf", clf)])

def train():
    os.makedirs(RES, exist_ok=True)
    df = pd.read_csv(DATA)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.language, test_size=0.2, stratify=df.language, random_state=42)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}  Languages: {df.language.nunique()}")

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.05),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=10),
        "Linear SVM": LinearSVC(C=1.0),
    }
    results, best_name, best_pipe, best_acc = {}, None, None, -1
    for name, clf in models.items():
        pipe = make_pipeline(clf)
        t = time.time(); pipe.fit(X_tr, y_tr); fit_t = time.time() - t
        pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "train_seconds": round(fit_t, 3)}
        print(f"{name:25s} acc={acc:.4f} macroF1={f1:.4f} fit={fit_t:.2f}s")
        if acc > best_acc: best_name, best_pipe, best_acc, best_pred = name, pipe, acc, pred

    print("\nBest:", best_name)
    report = classification_report(y_te, best_pred, digits=3, output_dict=True, zero_division=0)
    print(classification_report(y_te, best_pred, digits=3, zero_division=0))
    json.dump({"models": results, "best": best_name, "per_class": report,
               "n_train": len(X_tr), "n_test": len(X_te)}, open(os.path.join(RES, "metrics.json"), "w"), indent=2)

    # confusion matrix
    labels = sorted(df.language.unique())
    cm = confusion_matrix(y_te, best_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center",
                                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix - {best_name}")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(RES, "confusion_matrix.png"), dpi=130); plt.close(fig)

    # model comparison chart
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = list(results); accs = [results[n]["accuracy"] * 100 for n in names]
    bars = ax.bar(names, accs, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(min(accs) - 5, 101); ax.set_ylabel("Test accuracy (%)")
    for b, a in zip(bars, accs): ax.text(b.get_x() + b.get_width() / 2, a + 0.3, f"{a:.1f}", ha="center")
    plt.xticks(rotation=10); fig.tight_layout(); fig.savefig(os.path.join(RES, "model_comparison.png"), dpi=130); plt.close(fig)

    # accuracy vs. input length
    lens = X_te.str.len().values; correct = (best_pred == y_te.values)
    bins = [0, 40, 60, 90, 1000]; names_b = ["<40", "40-60", "60-90", ">90"]
    accs_b = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (lens >= lo) & (lens < hi); accs_b.append(correct[m].mean() * 100 if m.any() else np.nan)
    fig, ax = plt.subplots(figsize=(5.5, 3.6)); ax.bar(names_b, accs_b, color="#72B7B2")
    ax.set_xlabel("Input length (characters)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "accuracy_by_length.png"), dpi=130); plt.close(fig)

    joblib.dump(best_pipe, MODEL)
    print("Saved", MODEL)

def predict(text, top=3):
    pipe = joblib.load(MODEL)
    clf = pipe.named_steps["clf"]
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba([text])[0]
    else:
        s = pipe.decision_function([text])[0]; e = np.exp(s - s.max()); p = e / e.sum()
    idx = np.argsort(p)[::-1][:top]
    return [(clf.classes_[i], float(p[i])) for i in idx]

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 3 and sys.argv[1] == "predict":
        for lang, p in predict(" ".join(sys.argv[2:])): print(f"{lang:12s} {p:.2%}")
    else:
        print(__doc__)
