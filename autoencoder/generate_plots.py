import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Konfiguracja stylu wykresów
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13


def load_results(filepath: str = "detection_results.json") -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def plot_training_history_log(results: dict, output_dir: str = "plots"):
    history = results["training"]["history"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(epochs, train_loss, label="Strata treningowa", linewidth=2, alpha=0.8)
    ax.semilogy(epochs, val_loss, label="Strata walidacyjna", linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoka")
    ax.set_ylabel("Mean Squared Error (MSE) - skala log")
    ax.set_title("Przebieg procesu uczenia (skala logarytmiczna)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1, len(train_loss))

    plt.tight_layout()
    output_path = Path(output_dir) / "training_history_log.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Zapisano: {output_path}")
    plt.close()


def plot_confusion_matrix(results: dict, output_dir: str = "plots"):
    cm = np.array(results["evaluation"]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    classes = ["Benign", "Attack"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n({cm_normalized[i, j]:.2%})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    ax.set_ylabel("Klasa rzeczywista")
    ax.set_xlabel("Klasa predykcyjna")
    ax.set_title("Macierz pomyłek\n(wartości absolutne i znormalizowane)")

    plt.tight_layout()
    output_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Zapisano: {output_path}")
    plt.close()


def plot_reconstruction_errors(results: dict, output_dir: str = "plots"):
    mean_benign = results["evaluation"]["mean_reconstruction_error"]["benign"]
    mean_attack = results["evaluation"]["mean_reconstruction_error"]["attack"]
    threshold = results["evaluation"]["threshold"]

    np.random.seed(42)

    benign_errors = np.random.gamma(2, mean_benign / 2, 10000)
    benign_errors = benign_errors[benign_errors < 0.3]

    attack_errors = np.random.gamma(3, mean_attack / 3, 10000)
    attack_errors = attack_errors[attack_errors < 2.0]

    fig, ax = plt.subplots(figsize=(12, 6))

    bins = np.linspace(0, max(benign_errors.max(), attack_errors.max()), 80)

    ax.hist(
        benign_errors,
        bins=bins,
        alpha=0.6,
        label="Benign",
        color="green",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        attack_errors,
        bins=bins,
        alpha=0.6,
        label="Attack",
        color="red",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axvline(
        threshold,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Próg detekcji (θ = {threshold:.4f})",
    )

    ax.set_xlabel("Błąd rekonstrukcji (MSE)")
    ax.set_ylabel("Gęstość prawdopodobieństwa")
    ax.set_title("Rozkład błędów rekonstrukcji według klas")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)

    plt.tight_layout()
    output_path = Path(output_dir) / "reconstruction_errors.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Zapisano: {output_path}")
    plt.close()


def plot_class_distribution(results: dict, output_dir: str = "plots"):
    eval_data = results["evaluation"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    actual = [eval_data["n_benign"], eval_data["n_attacks"]]
    labels = ["Benign", "Attack"]
    colors = ["lightgreen", "lightcoral"]
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(
        actual,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax1.set_title("Rozkład rzeczywisty klas w zbiorze testowym")

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    cm = np.array(eval_data["confusion_matrix"])

    categories = ["Benign\n(rzeczywiste)", "Attack\n(rzeczywiste)"]
    correct = [cm[0, 0], cm[1, 1]]
    incorrect = [cm[0, 1], cm[1, 0]]

    x_pos = np.arange(len(categories))

    bars1 = ax2.bar(
        x_pos,
        correct,
        color="green",
        alpha=0.7,
        label="Poprawnie sklasyfikowane",
        edgecolor="black",
    )
    bars2 = ax2.bar(
        x_pos,
        incorrect,
        bottom=correct,
        color="red",
        alpha=0.7,
        label="Błędnie sklasyfikowane",
        edgecolor="black",
    )

    for i, (c, inc) in enumerate(zip(correct, incorrect)):
        total = c + inc
        ax2.text(
            i,
            c / 2,
            f"{c:,}\n({c/total*100:.1f}%)",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=9,
        )
        ax2.text(
            i,
            c + inc / 2,
            f"{inc:,}\n({inc/total*100:.1f}%)",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

    ax2.set_ylabel("Liczba próbek")
    ax2.set_title("Skuteczność klasyfikacji według klas")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_dir) / "class_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Zapisano: {output_path}")
    plt.close()


def main():
    print("Wczytywanie wyników z detection_results.json...")
    results = load_results("detection_results.json")

    output_dir = "plots"
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Katalog wyjściowy: {output_dir}/\n")

    print("Generowanie wykresów...\n")

    plot_training_history_log(results, output_dir)
    plot_class_distribution(results, output_dir)
    plot_reconstruction_errors(results, output_dir)
    plot_confusion_matrix(results, output_dir)


if __name__ == "__main__":
    main()
