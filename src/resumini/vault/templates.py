MATERIA_INDEX = """---
materia: {nombre}
type: index
cuatrimestre: {cuatrimestre}
profesor: {profesor}
tags:
  - {nombre}
---

# {nombre}

- **Profesor**: {profesor}
- **Cuatrimestre**: {cuatrimestre}
- **Fuentes**: {fuentes}"""

RESUMEN = """---
materia: {materia}
type: summary
source: {source}
date: {date}
tags:
  - {materia}
  - resumen
---

# {titulo}

## Ideas principales

{ideas}

## Conceptos clave

{conceptos}

## Conexiones

{conexiones}

## Notas

{notas}"""

APUNTE_CLASE = """---
materia: {materia}
type: apunte
fecha: {fecha}
tags:
  - {materia}
  - apunte
---

# Apunte de clase — {fecha}

## Tema

{tema}

## Apuntes

{apuntes}

## Dudas

{dudas}"""

PROFILE = """---
type: profile
---

# Perfil de estudio

## Preferencias de estilo

- **Formato preferido**: {formato}
- **Nivel de detalle**: {detalle}

## Materias activas

{materias}

## Notas sobre mi estilo

{notas}"""
