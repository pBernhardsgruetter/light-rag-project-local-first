# GraphRAG System mit LightRAG & KuzuDB

Autonomes GraphRAG-System für Wissensgraphen-Konstruktion, Dual-Level Retrieval und kontinuierliche Qualitätsvalidierung.

## 🏗️ Architekturskizze

```
                               ┌─────────────────┐
                               │  Rohdokumente   │
                               │  (data/input)   │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │ Extractor (GPU) │
                               │ GLiNER + REBEL  │
                               └────────┬────────┘
                                        │
                                        ▼
┌─────────────────┐            ┌─────────────────┐
│ LightRAG Service├───────────►│ KuzuDB Graph DB │
│ (Dual Retrieval)│            │  (Cypher ACID)  │
└────────┬────────┘            └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Validator Loop  │
│ LLM-as-a-Judge  │
└─────────────────┘
```

## 🚀 Schnellstart

### 1. Umgebungsvariablen einrichten
```bash
cp .env.example .env
# Editieren Sie .env und tragen Sie Ihren OPENROUTER_API_KEY ein
```

### 2. Docker Stack starten
```bash
docker compose build
docker compose up -d
```

### 3. Dokumente verarbeiten
Legen Sie beliebige PDF-, TXT- oder MD-Dateien in das Verzeichnis `data/input/`. Der Extractor verarbeitet diese automatisch inkrementell.

### 4. Retrieval-Abfragen testen
```bash
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Wie hängen EV-Adoption und Luftqualität zusammen?", "mode": "hybrid"}'
```

### 5. Migration auf lokales LLM (ThinkingCap)
Sobald Ihre lokale GPU-Infrastruktur bereitsteht:
```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d lightrag ollama
```

## 📄 Lizenz
MIT License
