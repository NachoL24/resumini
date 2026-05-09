# Resumini — Agente de Resumenes para la Facultad

## Overview

Resumini es un agente con gestion de memoria propia que ayuda a procesar material de estudio (PDFs, videos, PPTs, apuntes de clase) y generar resumenes personalizados para 8-10 materias universitarias. El sistema aprende el estilo de estudio del usuario, conecta conceptos entre materias, y exporta resumenes a Notion o Google Drive.

## Requisitos

- **Personalizacion moderada**: el agente aprende el estilo del usuario y conecta conceptos entre fuentes
- **Memoria persistente**: recuerda preferencias, materias activas, y contenido procesado entre sesiones
- **Destino flexible**: one-way sync a Notion y Google Drive (fase 1), bidireccional (fase 6)
- **Interfaz dual**: Telegram + CLI local
- **API paga**: sin restricciones de presupuesto para LLMs
- **Escala**: 8-10 materias, libros enteros, decenas de videos
- **Usuario**: alto nivel tecnico, puede deployar y mantener servicios

## Arquitectura

```
┌─────────────┐  ┌─────────────┐
│  Telegram   │  │  CLI Local  │
│    Bot      │  │  (stdin)    │
└──────┬──────┘  └──────┬──────┘
       │                │
       └───────┬────────┘
               ▼
       ┌───────────────┐
       │  API Gateway  │  (FastAPI)
       │    + Auth     │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │  Agent Core   │  (LangGraph)
       │  - Orchestrate│
       │  - Route tools│
       │  - Memory mgmt│
       └───────┬───────┘
               │
   ┌──────────┼──────────┬──────────────┐
   ▼          ▼          ▼              ▼
┌────────┐┌────────┐┌─────────┐  ┌──────────┐
│Ingest  ││Resume  ││Memory   │  │Export    │
│Tools   ││Tools   ││Tools    │  │Tools     │
│- PDF   ││- Gen   ││- Read   │  │- Notion  │
│- Video ││- Edit  ││- Write  │  │- Drive   │
│- Audio ││- Merge ││- Search │  │          │
└────┬───┘└────┬───┘└────┬────┘  └──────────┘
     │         │         │
     └─────────┼─────────┘
               ▼
       ┌───────────────┐
       │   Data Layer  │
       │ - Vault (.md) │  ← Fuente de verdad
       │ - ChromaDB    │  ← Indice vectorial
       │ - SQLite      │  ← Metadata/sessions
       └───────────────┘
```

### Principios

- El **vault de markdown** es la fuente de verdad — todo resumen vive como archivo `.md`
- ChromaDB es solo un **indice** para busqueda semantica (se puede reconstruir del vault)
- SQLite guarda metadata (materias, sesiones, preferencias de usuario)
- Las exportaciones a Notion/Drive son **one-way sync** en fase 1 (vault → destino)

## Estructura del Vault

```
vault/
├── profile.md                  ← Perfil de estudio, preferencias, estilo
├── materias/
│   ├── derecho_civil/
│   │   ├── _index.md           ← Metadata (profesor, cuatrimestre, fuentes)
│   │   ├── unidad_1.md
│   │   ├── unidad_2.md
│   │   ├── clase_2025-04-15.md ← Apuntes de clase
│   │   └── resumen_parcial.md
│   ├── penal/
│   │   └── ...
│   └── ...
├── templates/
│   ├── materia.md              ← Template para nueva materia
│   ├── resumen.md              ← Template para nuevo resumen
│   └── apunte_clase.md         ← Template para apunte de clase
└── .meta/
    └── sessions.json           ← Historial de sesiones
```

### Modelo de Memoria

- **Perfil**: el agente lee `profile.md` al inicio de cada sesion para recordar estilo, preferencias, materias activas
- **Resumenes**: cada archivo `.md` es editable por el usuario y por el agente
- **Indexado**: cada vez que el agente crea/modifica un `.md`, lo indexa en ChromaDB automaticamente
- **Cross-materia**: ChromaDB permite buscar conceptos similares entre materias

## Flujo del Agente (LangGraph)

```
┌─────────┐
│  INPUT   │  (mensaje de Telegram o CLI)
└────┬────┘
     ▼
┌─────────┐
│  ROUTER  │  Clasifica la intencion:
└────┬────┘  - Ingestar material
     │      - Generar resumen
     │      - Pregunta sobre contenido
     │      - Editar resumen
     │
┌────┼────┬────┬────┐
▼    ▼    ▼    ▼    ▼
┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
│INGEST││SUMMAR││QUERY ││ EDIT ││ SYNC │
│ NODE ││ IZE  ││ RAG  ││ NODE ││ NODE │
└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬──┘
   │       │       │       │       │
   └───────┴───────┴───────┘       │
              ▼                    │
       ┌──────────┐                │
       │ VALIDATE │  Revisa calidad del output
       │ + FORMAT │  (completo? bien formateado?)
       └────┬─────┘
            ▼
       ┌──────────┐
       │ MEMORY   │  Actualiza vault + ChromaDB
       │ UPDATE   │
       └────┬─────┘
            ▼
       ┌──────────┐
       │ EXPORT   │  Notion/Drive (one-way, fase 1)
       │ SYNC     │
       └────┬─────┘
            ▼
       ┌──────────┐
       │ RESPONSE │  Respuesta al usuario
       │ NODE     │
       └──────────┘
```

### Nodos

- **Router**: usa el LLM para clasificar la intencion del usuario (no regex)
- **Ingest node**: procesa PDF (PyMuPDF), video (Whisper API), audio, PPT (python-pptx). Extrae texto y lo guarda como `.md` crudo + lo indexa en ChromaDB
- **Summarize node**: lee contenido relevante del vault + ChromaDB, genera resumen con el estilo del usuario (definido en `profile.md`), lo escribe al vault
- **Query/RAG node**: busca en ChromaDB por similitud semantica, cruza con el vault para contexto, responde
- **Edit node**: modifica un resumen existente basado en instrucciones del usuario
- **Sync node**: exporta resumenes a Notion/Drive
- **Validate**: check de que el output no este vacio y tenga la estructura correcta
- **Memory update**: escribe/actualiza `.md` en vault, re-indexa en ChromaDB, actualiza `sessions.json`

## Stack Tecnico

| Componente | Tecnologia | Justificacion |
|---|---|---|
| Agente framework | LangGraph (Python) | Control fino del grafo, buena doc, ecosistema LangChain |
| API | FastAPI | Rapido, async, soporta Telegram webhooks + CLI |
| Vector DB | ChromaDB (embedded) | Simple, local, no necesita servidor separado |
| Metadata DB | SQLite | Local, simple, para sesiones y preferencias |
| Vault | Archivos .md en filesystem | Transparente, editable, Obsidian-compatible |
| Telegram | python-telegram-bot | Maduro, async, soporta archivos/media |
| PDF parsing | PyMuPDF (fitz) | Rapido, maneja PDFs complejos con imagenes |
| Video/Audio | OpenAI Whisper API | Transcripcion de calidad sin modelo local |
| PPT parsing | python-pptx | Standard para PowerPoints |
| Notion API | notion-client | Oficial, soporta pages/blocks/search |
| Google Drive | google-api-python-client | Oficial, docs/sheets/drive |
| LLM | Claude/GPT-4o via API | Razonamiento y generacion de resumenes |
| Embeddings | OpenAI text-embedding-3-small | Barato, rapido, buen quality para RAG |
| Deployment | Docker Compose | Un comando para levantar todo |

## Fases de Implementacion

| Fase | Alcance | Tiempo estimado |
|---|---|---|
| F1 - MVP | API + agente basico (ingesta PDF + resumen) + vault + ChromaDB + CLI | 1-2 semanas |
| F2 - Telegram | Bot de Telegram con envio de archivos + ingesta de media | 3-5 dias |
| F3 - Export | One-way sync a Notion + Google Drive | 3-5 dias |
| F4 - Video/Audio | Transcripcion con Whisper + ingesta de videos/audios | 2-3 dias |
| F5 - Cross-materia | Busqueda semantica entre materias, conexion de conceptos | 2-3 dias |
| F6 - Bidirectional | Sync bidireccional Notion/Drive (webhooks/polling) | 1 semana |

## Decisiones Abiertas

- **Formato de resumen por defecto**: definir en `profile.md` templates de como se estructura un resumen (headings, bullets, highlights)
- **Modelo LLM primario**: Claude vs GPT-4o — definir segun calidad de resumen y costo por token en contexto largo
- **Estrategia de chunking para ChromaDB**: por secciones de markdown (heading-based) vs por cantidad fija de tokens
- **Limites de contexto**: como manejar libros enteros (500+ paginas) — chunking progresivo con resumenes jerarquicos
