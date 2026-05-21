# Download Ontology Files

Download these three files before running any scripts. All files go into `data/ontologies/` (relative to the repo root, one level above `exclusively-neo4j/`).

---

## File 1: doid.obo — Disease Ontology

**URL:** https://purl.obolibrary.org/obo/doid.obo
**Save to:** `data/ontologies/doid.obo`
**Size:** approximately 50 MB

---

## File 2: hp.obo — Human Phenotype Ontology

**URL:** https://purl.obolibrary.org/obo/hp.obo
**Save to:** `data/ontologies/hp.obo`
**Size:** approximately 30 MB

---

## File 3: phenotype.hpoa — HPO Disease-Phenotype Annotations

**URL:** https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa
**Save to:** `data/ontologies/phenotype.hpoa`
**Size:** approximately 100 MB

---

## Download commands

### PowerShell (Windows)

```powershell
New-Item -ItemType Directory -Force -Path ..\data\ontologies

Invoke-WebRequest -Uri "https://purl.obolibrary.org/obo/doid.obo" `
    -OutFile "..\data\ontologies\doid.obo"

Invoke-WebRequest -Uri "https://purl.obolibrary.org/obo/hp.obo" `
    -OutFile "..\data\ontologies\hp.obo"

Invoke-WebRequest -Uri "https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa" `
    -OutFile "..\data\ontologies\phenotype.hpoa"
```

### curl (Linux / macOS)

```bash
mkdir -p ../data/ontologies
curl -L https://purl.obolibrary.org/obo/doid.obo \
     -o ../data/ontologies/doid.obo
curl -L https://purl.obolibrary.org/obo/hp.obo \
     -o ../data/ontologies/hp.obo
curl -L https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa \
     -o ../data/ontologies/phenotype.hpoa
```

---

Once all three files are present, proceed to `2_parse_ontologies.py`.
