#  model/evaluate.py (English Output Version)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Ensure English-compatible font

def evaluate_classifier(y_true, y_prob, threshold=0.5):
    """
    Print classification metrics, confusion matrix, ROC curve.
    """
    y_pred = (y_prob > threshold).astype(int)

    print("📋 Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC Score: {round(auc, 4)}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Down', 'Pred: Up'],
                yticklabels=['True: Down', 'True: Up'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {round(auc, 4)}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
