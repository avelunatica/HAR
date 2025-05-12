import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non interactivo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from loguru import logger

def generate_report(
    csv_path: str,
    output_dir: str = "",
    separator: str = ","
):
    """Xera informe de métricas a partir dun CSV de predicións."""

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- AXUSTES GLOBAIS ---
    sns.set_context("talk")
    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
    })

    logger.info(f"📄 Lendo CSV de predicións: {csv_path}")
    df = pd.read_csv(csv_path, sep=separator)
    df.columns = df.columns.str.strip()

    labels = [
        "drinking", "eating something", "brushing teeth",
        "using computer", "writing or painting/drawing"
        
    ]

    label_map = {
        "drinking": "beber",
        "eating something": "comer algo",
        "brushing teeth": "cepillarse os dentes",
        "using computer": "usar o ordenador",
        "writing or painting/drawing": "escribir ou pintar/dibuxar",
    }

    df["True Class"] = df["True Class"].astype(str)
    df["Predicted Class"] = df["Predicted Class"].astype(str)
    y_true = df["True Class"]
    y_pred = df["Predicted Class"]

    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro") * 100
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted") * 100

    logger.success(f" F1‑macro: {f1_macro:.2f}% | F1‑ponderado: {f1_weighted:.2f}%")

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=[label_map[l] for l in labels],
        output_dict=True,
        zero_division=0
    )
    df_report = pd.DataFrame(report).T
    df_report = df_report.reindex([*label_map.values()] + ['accuracy', 'macro avg', 'weighted avg'])

    logger.info("Xerando matriz de confusión...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10), facecolor='none')
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[label_map[l] for l in labels],
        yticklabels=[label_map[l] for l in labels],
        cbar=False, linewidths=.5, linecolor='gray'
    ).patch.set_alpha(0)
    plt.xlabel("Clase Predita")
    plt.ylabel("Clase Real")
    plt.title("Matriz de confusión por acción")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), transparent=True)
    plt.close()

    logger.success(f" Matriz gardada en {plots_dir}/confusion_matrix.png")

    logger.info("Xerando gráfico F1-score por clase...")
    f1_per_class = df_report.loc[label_map.values(), 'f1-score'] * 100
    plt.figure(figsize=(12, 6), facecolor='none')
    sns.barplot(
        x=list(f1_per_class.values),
        y=list(f1_per_class.index),
        palette="viridis"
    ).patch.set_alpha(0)
    plt.xlabel("Puntuación F1 (%)")
    plt.ylabel("Acción")
    plt.title("Puntuación F1 por clase")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "f1score_por_clase.png"), transparent=True)
    plt.close()

    logger.success(f"Gráfica gardada en {plots_dir}/f1score_por_clase.png")

    logger.info("Exportando resumo de métricas...")
    summary = df_report.loc[label_map.values(), ['precision', 'recall', 'f1-score']].copy()
    summary.columns = ['precision', 'recall', 'f1_score']
    summary['samples'] = df.groupby("True Class")["True Class"].count().loc[labels].values
    summary.index.name = 'acción'
    summary.to_csv(os.path.join(output_dir, "resumo_f1score_por_clase.csv"), float_format="%.2f")

    logger.success(f"Resumo exportado en {output_dir}/resumo_f1score_por_clase.csv")

if __name__ == "__main__":
    import fire
    fire.Fire(generate_report)
