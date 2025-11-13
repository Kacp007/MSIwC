import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class DatasetSnapshot:
    row_count: int
    column_count: int
    columns: Dict[str, str]


@dataclass
class ProcessingSummary:
    source_description: str
    snapshot_before: DatasetSnapshot
    snapshot_after: DatasetSnapshot
    dropped_duplicates: int
    dropped_constant_columns: List[str]
    dropped_high_na_columns: List[str]
    filled_missing_columns: Dict[str, str]
    replaced_infinite_values: int
    scaling_method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skrypt do wstępnego przetwarzania i normalizacji danych sieciowych."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Ścieżka do wejściowego pliku CSV z danymi lub katalogu z plikami CSV.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Ścieżka do katalogu z plikami CSV do przetworzenia (alternatywa dla input_path).",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=None,
        help=(
            "Ścieżka do zapisu znormalizowanego zbioru danych (CSV). "
            "Jeśli nie podano, zostanie użyty folder 'output' i nazwa pliku z sufiksem '-normalized'."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Ścieżka do raportu opisującego przygotowanie danych (Markdown).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Ścieżka do zapisu metadanych przetwarzania w formacie JSON.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Krótki opis pochodzenia danych wykorzystany w raporcie.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.6,
        help=(
            "Maksymalny dopuszczalny udział braków danych w kolumnie (0-1). "
            "Kolumny z większym udziałem zostaną usunięte."
        ),
    )
    parser.add_argument(
        "--scaling",
        choices=["standard", "minmax"],
        default="standard",
        help="Metoda skalowania wartości numerycznych.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Opcjonalna liczba rekordów do załadowania (losowa próbka).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Ziarno generatora losowego dla próbkowania.",
    )
    return parser.parse_args()


def load_dataset(path: Path, sample: Optional[int], random_state: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=random_state)
    return df.reset_index(drop=True)


def load_all_datasets(data_dir: Path, sample: Optional[int], random_state: int) -> pd.DataFrame:
    """Ładuje wszystkie pliki CSV z katalogu i łączy je w jeden DataFrame."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nie znaleziono plików CSV w katalogu: {data_dir}")
    
    print(f"Znaleziono {len(csv_files)} plików CSV do przetworzenia:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    dataframes = []
    for csv_file in csv_files:
        print(f"Ładowanie {csv_file.name}...")
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Połączono {len(dataframes)} plików. Łączna liczba rekordów: {len(combined_df)}")
    
    if sample is not None and sample < len(combined_df):
        combined_df = combined_df.sample(n=sample, random_state=random_state)
        print(f"Wylosowano próbkę: {len(combined_df)} rekordów")
    
    return combined_df.reset_index(drop=True)


def create_snapshot(df: pd.DataFrame) -> DatasetSnapshot:
    column_descriptions: Dict[str, str] = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        descriptor_parts: List[str] = [dtype]
        col_lower = col.lower()
        hints: List[str] = []
        if "ip" in col_lower:
            hints.append("możliwy adres IP")
        if "time" in col_lower or "timestamp" in col_lower:
            hints.append("znacznik czasu")
        if "packet" in col_lower or "pkt" in col_lower:
            hints.append("wielkość/cecha pakietu")
        if "attack" in col_lower or "label" in col_lower:
            hints.append("etykieta ataku/ruchu normalnego")
        if hints:
            descriptor_parts.append(f"({', '.join(hints)})")
        column_descriptions[col] = " ".join(descriptor_parts)
    return DatasetSnapshot(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        columns=column_descriptions,
    )


def drop_constant_columns(df: pd.DataFrame) -> List[str]:
    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    df.drop(columns=constant_columns, inplace=True)
    return constant_columns


def drop_high_na_columns(df: pd.DataFrame, threshold: float) -> List[str]:
    na_ratio = df.isna().mean()
    to_drop = na_ratio[na_ratio > threshold].index.tolist()
    df.drop(columns=to_drop, inplace=True)
    return to_drop


def clean_missing_values(df: pd.DataFrame) -> Dict[str, str]:
    strategies: Dict[str, str] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            fill_value = df[col].median()
            df.loc[:, col] = df[col].fillna(fill_value)
            strategies[col] = "uzupełniono medianą"
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    for col in categorical_cols:
        if df[col].isna().any():
            fill_value = df[col].mode(dropna=True)
            if len(fill_value) > 0:
                df.loc[:, col] = df[col].fillna(fill_value.iloc[0])
                strategies[col] = "uzupełniono dominantą"
            else:
                df.loc[:, col] = df[col].fillna("brak")
                strategies[col] = "uzupełniono stałą 'brak'"
    return strategies


def replace_infinite_values(df: pd.DataFrame) -> int:
    mask = ~np.isfinite(df.select_dtypes(include=[np.number]))
    if mask.empty:
        return 0
    count = int(mask.to_numpy().sum())
    if count > 0:
        df.mask(mask, np.nan, inplace=True)
    return count


def scale_numeric_columns(df: pd.DataFrame, method: str) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


def generate_report(summary: ProcessingSummary, path: Path, label_counts: Optional[Dict[str, int]] = None) -> None:
    lines: List[str] = []
    lines.append("# Przygotowanie zbioru danych (normalizacja)\n")

    lines.append("## 1. Charakterystyka danych\n")
    lines.append(
        f"- Liczba rekordów (przed czyszczeniem): {summary.snapshot_before.row_count}"
    )
    lines.append(
        f"- Liczba kolumn (przed czyszczeniem): {summary.snapshot_before.column_count}"
    )
    lines.append("- Typy i charakterystyka kolumn:")
    for name, description in summary.snapshot_before.columns.items():
        lines.append(f"  - `{name}`: {description}")
    lines.append("")
    
    if label_counts:
        lines.append("### Rozkład etykiet ataków\n")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / summary.snapshot_before.row_count) * 100
            lines.append(f"- `{label}`: {count} ({percentage:.2f}%)")
        lines.append("")

    lines.append("## 2. Źródło danych\n")
    lines.append(f"{summary.source_description}\n")

    lines.append("## 3. Wstępne przetwarzanie danych\n")
    lines.append(
        f"- Usunięto duplikaty: {summary.dropped_duplicates} rekordów mniej."
    )
    if summary.dropped_constant_columns:
        lines.append(
            "- Usunięto kolumny o stałych wartościach: "
            + ", ".join(f"`{col}`" for col in summary.dropped_constant_columns)
        )
    else:
        lines.append("- Nie stwierdzono kolumn o stałych wartościach.")
    if summary.dropped_high_na_columns:
        lines.append(
            "- Usunięto kolumny z nadmiarem braków danych: "
            + ", ".join(f"`{col}`" for col in summary.dropped_high_na_columns)
        )
    else:
        lines.append("- Nie usuwano kolumn z powodu braków danych.")
    if summary.filled_missing_columns:
        lines.append("- Strategie uzupełniania braków danych:")
        for col, strategy in summary.filled_missing_columns.items():
            lines.append(f"  - `{col}`: {strategy}")
    else:
        lines.append("- W danych nie występowały braki wymagające uzupełnienia.")
    if summary.replaced_infinite_values > 0:
        lines.append(
            f"- Zastąpiono wartości nieskończone: {summary.replaced_infinite_values} → NaN."
        )
    lines.append(f"- Zastosowana metoda skalowania: {summary.scaling_method}.")
    lines.append("")

    lines.append("## 5. Wyniki i wnioski z przygotowania danych\n")
    lines.append(
        f"- Liczba rekordów po przetwarzaniu: {summary.snapshot_after.row_count}"
    )
    lines.append(
        f"- Liczba kolumn po przetwarzaniu: {summary.snapshot_after.column_count}"
    )
    reduction_rows = summary.snapshot_before.row_count - summary.snapshot_after.row_count
    reduction_cols = summary.snapshot_before.column_count - summary.snapshot_after.column_count
    lines.append(
        f"- Zmniejszenie liczby rekordów: {reduction_rows} ("
        f"{_format_percentage(reduction_rows, summary.snapshot_before.row_count)})"
    )
    lines.append(
        f"- Zmniejszenie liczby kolumn: {reduction_cols} ("
        f"{_format_percentage(reduction_cols, summary.snapshot_before.column_count)})"
    )
    lines.append(
        "- Normalizacja wartości numerycznych powinna skrócić czas uczenia modeli "
        "i ułatwić konwergencję algorytmów statystycznych oraz metod uczenia maszynowego."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _format_percentage(value: int, base: int) -> str:
    if base == 0:
        return "brak danych"
    return f"{(value / base) * 100:.2f}%"


def save_summary_json(summary: ProcessingSummary, path: Path) -> None:
    data = asdict(summary)
    data["snapshot_before"]["columns"] = dict(summary.snapshot_before.columns)
    data["snapshot_after"]["columns"] = dict(summary.snapshot_after.columns)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    # Określenie czy przetwarzamy katalog czy pojedynczy plik
    if args.input_dir is not None:
        if not args.input_dir.exists() or not args.input_dir.is_dir():
            raise FileNotFoundError(f"Nie znaleziono katalogu: {args.input_dir}")
        data_dir = args.input_dir
        df_raw = load_all_datasets(data_dir, args.sample, args.random_state)
    elif args.input_path.is_dir():
        data_dir = args.input_path
        df_raw = load_all_datasets(data_dir, args.sample, args.random_state)
    else:
        if not args.input_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {args.input_path}")
        df_raw = load_dataset(args.input_path, args.sample, args.random_state)
    
    snapshot_before = create_snapshot(df_raw)

    df_processed = df_raw.copy()
    duplicates_before = df_processed.shape[0]
    df_processed.drop_duplicates(inplace=True)
    dropped_duplicates = duplicates_before - df_processed.shape[0]

    dropped_constant_columns = drop_constant_columns(df_processed)
    dropped_high_na_columns = drop_high_na_columns(df_processed, args.missing_threshold)
    replaced_infinite_values = replace_infinite_values(df_processed)
    filled_missing_columns = clean_missing_values(df_processed)
    scale_numeric_columns(df_processed, args.scaling)

    snapshot_after = create_snapshot(df_processed)
    
    # Zliczenie etykiet dla raportu (przed przetwarzaniem, aby pokazać oryginalny rozkład)
    label_counts = None
    if " Label" in df_raw.columns:
        label_counts = df_raw[" Label"].value_counts().to_dict()

    # Określenie ścieżki wyjściowej
    if args.output_data is not None:
        output_data_path = args.output_data
    elif args.input_dir is not None or (args.input_path.is_dir() if args.input_path.exists() else False):
        output_data_path = Path("output") / "combined-normalized.csv"
    else:
        output_data_path = Path("output") / f"{args.input_path.stem}-normalized.csv"

    # Określenie opisu źródła danych
    if args.data_source is not None:
        source_description = args.data_source
    elif args.input_dir is not None or (args.input_path.is_dir() if args.input_path.exists() else False):
        source_description = "Źródło danych: CIC-IDS2017 (zbiór połączony - DDoS, FTP-BruteForce i inne ataki)."
    else:
        source_description = "Źródło danych nie zostało określone przez użytkownika."
    
    summary = ProcessingSummary(
        source_description=source_description,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        dropped_duplicates=int(dropped_duplicates),
        dropped_constant_columns=dropped_constant_columns,
        dropped_high_na_columns=dropped_high_na_columns,
        filled_missing_columns=filled_missing_columns,
        replaced_infinite_values=replaced_infinite_values,
        scaling_method="StandardScaler" if args.scaling == "standard" else "MinMaxScaler",
    )

    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_data_path, index=False)

    # Określenie ścieżek dla raportów (jeden zbiorczy raport)
    report_path = args.report if args.report is not None else Path("reports") / "combined-preprocessing-report.md"
    summary_json_path = args.summary_json if args.summary_json is not None else Path("reports") / "combined-preprocessing-summary.json"
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(summary, report_path, label_counts)

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary_json(summary, summary_json_path)

    print(f"Zapisano znormalizowane dane do: {output_data_path}")
    print(f"Raport z przetwarzania: {report_path}")
    print(f"Metadane procesu: {summary_json_path}")


if __name__ == "__main__":
    main()

