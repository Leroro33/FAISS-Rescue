#!/usr/bin/env python3
"""
02_download_genomes.py

Downloads 15 complete genomes per species from NCBI RefSeq
for 11 clinically relevant bacterial species using NCBI eutils API.

Outputs:
    data/raw_genomes/*.fna.gz
    data/raw_genomes/*_assembly_report.txt
"""

import urllib.request
import json
import time
import subprocess
import os

TAXIDS = {
    562:   "Escherichia coli",
    1280:  "Staphylococcus aureus",
    573:   "Klebsiella pneumoniae",
    28901: "Salmonella enterica",
    1639:  "Listeria monocytogenes",
    1313:  "Streptococcus pneumoniae",
    1314:  "Streptococcus pyogenes",
    1311:  "Streptococcus agalactiae",
    1773:  "Mycobacterium tuberculosis",
    287:   "Pseudomonas aeruginosa",
    470:   "Acinetobacter baumannii",
}

LIMIT   = 15
OUT_DIR = "data/raw_genomes"


def fetch_accessions(taxid, limit=15):
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=assembly&term=txid{taxid}[Organism]"
        f"+AND+%22Complete+Genome%22[AssemblyStatus]"
        f"+AND+latest_refseq[filter]&retmax={limit}&retmode=json"
    )
    with urllib.request.urlopen(url) as r:
        ids = json.loads(r.read())["esearchresult"]["idlist"]

    accessions = []
    for uid in ids:
        url2 = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=assembly&id={uid}&retmode=json"
        )
        with urllib.request.urlopen(url2) as r:
            data = json.loads(r.read())
            acc  = data["result"][uid]["assemblyaccession"]
            name = data["result"][uid]["speciesname"]
            accessions.append(acc)
            print(f"    {acc}  {name}")
        time.sleep(0.4)
    return accessions


def count_genomes():
    result = subprocess.run(
        ["find", OUT_DIR, "-name", "*.fna.gz"],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split("\n") if l]
    return len(lines)


def main():
    print("=== 02_download_genomes.py ===\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    all_accessions = []
    total = len(TAXIDS)

    for i, (taxid, name) in enumerate(TAXIDS.items()):
        print(f"[{i+1}/{total}] Fetching accessions for {name} (taxid: {taxid})...")
        accs = fetch_accessions(taxid, LIMIT)
        all_accessions.extend(accs)
        print(f"    Got {len(accs)} accessions\n")

    print(f"Total accessions to download: {len(all_accessions)}")

    cmd = [
        "ncbi-genome-download",
        "--assembly-accessions", ",".join(all_accessions),
        "--formats", "fasta,assembly-report",
        "--flat-output",
        "--output-folder", OUT_DIR,
        "--progress-bar",
        "bacteria"
    ]
    subprocess.run(cmd, check=True)

    print("\n=== DOWNLOAD COMPLETE ===")
    print(f"Total genomes:  {count_genomes()}")
    print(f"Output folder:  {OUT_DIR}")


if __name__ == "__main__":
    main()
