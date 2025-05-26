#!/usr/bin/env python3
from __future__ import annotations

"""
labparser.py − Converte file .lab (SpectraLAB) in CSV
e genera un PDF con i grafici degli spettri.

USO:
    python labparser.py <percorso_input> [-o DIR_OUTPUT]

• <percorso_input> può essere un singolo file .lab o una cartella che li contiene.
• Se -o/--output non è indicato, i CSV e il PDF vengono creati nella stessa cartella dei .lab.
• Dipendenze: pandas, matplotlib  (pip install pandas matplotlib).
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json


# -----------------------------------------------------------------------------#
# Utility: auto-detect encoding                                                #
# -----------------------------------------------------------------------------#
def read_text_auto(path: Path) -> str:
    """
    Legge un file tentando di determinare l’encoding.

    • UTF-16 (LE/BE) e UTF-8 con BOM vengono gestiti esplicitamente.
    • Negli altri casi si assume UTF-8 senza BOM (con sostituzione errori).
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16")        # UTF-16 little-endian BOM
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")     # UTF-16 big-endian BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")     # UTF-8 BOM
    return raw.decode("utf-8", errors="replace")


# -----------------------------------------------------------------------------#
# Parsing dei vettori                                                          #
# -----------------------------------------------------------------------------#
def parse_lab_text(text: str) -> dict[str, list[float]]:
    """
    Restituisce un dict {nome_vettore: lista_valori_float}.
    Implementa la stessa logica della funzione JavaScript `parseLab`.
    """
    vecs: dict[str, list[float]] = {}
    current_name: str | None = None
    collecting = False
    buffer: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()

        # Nuova sezione [vecteur] / [vector]
        if re.match(r"\[(?:vecteur|vector)\]", line, flags=re.I):
            current_name = None
            collecting = False
            buffer.clear()
            continue

        # Nome del vettore  nom="..."  |  name="..."
        m_name = re.match(r'(?:nom|name)\s*=\s*"([^"]+)"', line, flags=re.I)
        if m_name:
            current_name = m_name.group(1)
            continue

        # Inizio tabella punti
        if re.match(r"points\s*=\s*table", line, flags=re.I) and current_name:
            collecting = True
            buffer.clear()
            continue

        # Fine tabella punti
        if collecting and line.startswith("}"):
            nums: list[float] = []
            for row in buffer:
                row = row.replace(",", ".")              # virgole → punti
                nums.extend(float(tok) for tok in re.split(r"\s+", row) if tok)
            vecs[current_name] = nums
            collecting = False
            continue

        # Riga di punti
        if collecting:
            buffer.append(line)

    return vecs


# -----------------------------------------------------------------------------#
# Scelta interattiva (o mantenimento) delle serie                              #
# -----------------------------------------------------------------------------#
def _select_series_interactive(
    vecs: dict[str, list[float]],
    wl_key: str,
    filename: str
) -> dict[str, list[float]]:
    """
    Se il file contiene più di una serie Y, chiede:
    • 0  → mantenere tutte le serie  (opzione di default con invio vuoto)
    • 1‑N → esportare solo la serie scelta
    In caso di “tutte”, offre di rinominare ogni serie.

    Restituisce un dict che include λ e le Y finali.
    """
    y_keys = [k for k in vecs.keys() if k != wl_key]
    if len(y_keys) <= 1:
        return vecs  # già singola

    print(f"\n{filename} contiene {len(y_keys)} serie di intensità:")
    print("  0) Mantieni tutte le serie")
    for i, k in enumerate(y_keys, 1):
        print(f"  {i}) {k}")

    sel = None
    while sel is None:
        try:
            choice = input(f"Scegli (0‑{len(y_keys)} | invio=0): ").strip() or "0"
            idx = int(choice)
            if 0 <= idx <= len(y_keys):
                sel = idx
            else:
                print("Numero fuori range.")
        except (ValueError, EOFError):
            print("Input non valido.")

    if sel == 0:
        # Opzione di rinomina
        rename = input("Vuoi rinominare le serie? (s/N): ").strip().lower() == "s"
        if rename:
            new_vecs: dict[str, list[float]] = {wl_key: vecs[wl_key]}
            for k in y_keys:
                new_name = ""
                while not new_name:
                    new_name = input(f"Nuovo nome per \"{k}\": ").strip()
                    if new_name in new_vecs or not new_name:
                        print("Nome non valido o duplicato.")
                        new_name = ""
                new_vecs[new_name] = vecs[k]
            print("→ Tutte le serie saranno esportate con i nuovi nomi.\n")
            return new_vecs
        print("→ Verranno esportate tutte le serie senza modifiche.\n")
        return vecs

    sel_key = y_keys[sel - 1]
    print(f"→ Verrà esportata la serie: {sel_key}\n")
    return {wl_key: vecs[wl_key], sel_key: vecs[sel_key]}


# -----------------------------------------------------------------------------#
# DataFrame e salvataggio                                                      #
# -----------------------------------------------------------------------------#
def build_dataframe(vecs: dict[str, list[float]]) -> pd.DataFrame:
    """
    Crea un DataFrame con la colonna lunghezze d’onda (λ/lambda) e una
    colonna per ciascuna serie Y. Se le serie hanno lunghezze diverse,
    quelle più corte vengono riempite con NaN.

    Vengono conservati tutti e soli i campioni con λ > 0 (quindi fino a ~713 nm nel tuo spettrometro).
    """
    if not vecs:
        raise ValueError("Nessun vettore trovato nel file .lab")

    # Trova la chiave per la lunghezza d'onda
    wl_key = next(
        (k for k in vecs if k.lower() in {"λ", "lambda", "wavelength"}),
        next(iter(vecs))
    )

    n = len(vecs[wl_key])
    data: dict[str, list[float]] = {wl_key: vecs[wl_key]}

    # Sincronizza la lunghezza di tutte le serie
    for k, v in vecs.items():
        if k == wl_key:
            continue
        data[k] = v + [float("nan")] * (n - len(v))

    df = pd.DataFrame(data)

    # Mantieni solo i punti con λ strettamente positivo (scarta zeri/falsi header)
    df = df[df[wl_key] > 0].reset_index(drop=True)

    return df


def convert_file(path: Path, out_dir: Path | None = None) -> Path | None:
    """
    Converte un singolo file .lab in CSV.
    Ritorna il Path del CSV creato o None se il file è vuoto/non valido.
    """
    text = read_text_auto(path)
    vecs = parse_lab_text(text)
    # Determina la chiave λ e, se necessario, chiedi quale serie Y esportare
    wl_key = next(
        (k for k in vecs if k.lower() in {"λ", "lambda", "wavelength"}),
        next(iter(vecs))
    )
    vecs = _select_series_interactive(vecs, wl_key, path.name)

    try:
        df = build_dataframe(vecs)
    except ValueError as err:
        print(f"⚠️  {path.name}: {err}")
        return None

    out_dir = out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.csv"
    df.to_csv(out_path, index=False)
    return out_path


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#
def main() -> None:
    parser = argparse.ArgumentParser(description="Converte file .lab in CSV.")
    parser.add_argument(
        "input",
        type=Path,
        help="File .lab o cartella contenente file .lab",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Cartella di destinazione dei CSV/PDF (facoltativa)",
    )
    args = parser.parse_args()

    csv_paths: list[Path] = []

    # — Raccolta dei file —
    if args.input.is_dir():
        files = sorted(args.input.glob("*.lab"))
        if not files:
            raise SystemExit("Nessun file .lab trovato nella cartella.")
    elif args.input.suffix.lower() == ".lab":
        files = [args.input]
    else:
        raise SystemExit("Il percorso indicato deve essere .lab o una cartella.")

    # — Conversione —
    for f in files:
        csv_path = convert_file(f, args.output)
        if csv_path:
            print(f"✔️  Creato: {csv_path}")
            csv_paths.append(csv_path)

    # — JSON con i titoli —
    # Cerca list.json nella cartella di input o accanto al file singolo
    list_path = None
    if args.input.is_dir():
        cand = args.input / "list.json"
        if cand.exists():
            list_path = cand
    else:
        cand = args.input.parent / "list.json"
        if cand.exists():
            list_path = cand

    if list_path:
        with open(list_path, "r", encoding="utf-8") as f:
            title_map = json.load(f)
        titles_out: dict[str, str] = {}
        for csv in csv_paths:
            lab_name = csv.with_suffix(".lab").name
            if lab_name in title_map:
                titles_out[csv.name] = title_map[lab_name]
        if titles_out:
            json_dir = args.output if args.output else csv_paths[0].parent
            json_path = json_dir / "titoli.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(titles_out, f, ensure_ascii=False, indent=2)
            print(f"🗂️  JSON titoli generato: {json_path}")
    else:
        print("⚠️  list.json non trovato: salto creazione titoli.json")

    # — PDF con i grafici —
    if csv_paths:
        pdf_dir = (
            args.output
            if args.output
            else (args.input if args.input.is_dir() else args.input.parent)
        )
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "spettri.pdf"

        with PdfPages(pdf_path) as pdf:
            for csv in csv_paths:
                df = pd.read_csv(csv)
                wl_key = df.columns[0]
                ax = df.plot(x=wl_key, y=df.columns[1:], figsize=(8, 4))
                ax.set_xlabel("Lunghezza d’onda (nm)")
                ax.set_ylabel("Intensità (a.u.)")
                ax.set_title(csv.stem)
                pdf.savefig(ax.figure)
                plt.close(ax.figure)

        print(f"📄  PDF generato: {pdf_path}")


if __name__ == "__main__":
    main()