# Proyecto Final · Derivados y Administración de Riesgos

**Equipo 7 · Productos Derivados y Administración de Riesgos Corporativos · EGADE Business School**
María Tamara López Romo · Daniel Alejandro Sánchez Aguirre · Michel Alexander Meléndez Salas

Fecha de corte de los datos de mercado: **6 de noviembre de 2026**.

Empresa: **Apple Inc. (AAPL, Nasdaq)**. Siguiendo el caso *Molson Coors Brewing Co.: Valuing a Convertible Bond* (2007), el
proyecto propone cómo Apple puede financiarse y cubrir su tesorería con opciones, futuros, swaps y un bono convertible:
volatilidad GARCH, Monte Carlo, árboles binomiales y Black-Scholes, convertible con call spread, cross-currency swap,
interest rate swap, T-lock y futuros, 18 estrategias de opciones sobre AAPL, oro y WTI, y operaciones en simuladores.

**Página del proyecto:** https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/

## Entregables

| # | Entregable | Carpeta | Abrir |
|---|---|---|---|
| 1 | Reporte ejecutivo (Word y PDF) | [`1_reporte_ejecutivo/`](1_reporte_ejecutivo/) | [PDF](1_reporte_ejecutivo/Reporte_ejecutivo_Derivados.pdf) · [Word](1_reporte_ejecutivo/Reporte_ejecutivo_Derivados.docx) |
| 2 | Dashboard (HTML con Plotly) | [`2_dashboard/`](2_dashboard/) | [Abrir dashboard](https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/2_dashboard/dashboard_ejecutivo.html) |
| 3 | Reporte reproducible (Rmd y HTML) | [`3_reporte_reproducible/`](3_reporte_reproducible/) | [Abrir HTML](https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/3_reporte_reproducible/Proyecto_Final_Derivados.html) · [Rmd](3_reporte_reproducible/Proyecto_Final_Derivados.Rmd) |
| 4 | Calculadora interactiva (bono + opción y call spread) | [`4_calculadora/`](4_calculadora/) | [Abrir calculadora](https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/4_calculadora/calculadora_convertible.html) |
| 5 | Código en Python (.py y notebooks con outputs) | [`5_python/`](5_python/) | [Valuación en Colab](https://colab.research.google.com/github/danielsanchez-ai/Derivados_ProyectoFinal_AAPL/blob/main/5_python/valuacion_bs_arbol_convertible.ipynb) · [GARCH en Colab](https://colab.research.google.com/github/danielsanchez-ai/Derivados_ProyectoFinal_AAPL/blob/main/5_python/garch_derivados_AAPL_Oro_WTI_6E.ipynb) |
| 6 | Uso de IA (prompts, código y correcciones) | [`6_uso_IA/`](6_uso_IA/) | [Resumen](6_uso_IA/uso_de_IA_Equipo7.pdf) · [Conversación con TecGPT](6_uso_IA/conversacion_TecGPT_con_notas.pdf) |
| 7 | Anexo de capturas (LSEG, Bloomberg, CME y más) | [`7_anexo_capturas/`](7_anexo_capturas/) | [PDF del anexo](7_anexo_capturas/Anexo_capturas_Equipo7.pdf) |
| 8 | Repositorio de GitHub | este repositorio | — |

Material adicional: [`presentacion/`](presentacion/) (14 diapositivas en PDF) y [`antecedentes/`](antecedentes/) (trabajos
previos del equipo con datos de LSEG: Tarea 3 de estrategias y griegas, y análisis GARCH de AAPL del 3 de noviembre).

## Contenido de cada carpeta

| Ruta | Qué es |
|---|---|
| `1_reporte_ejecutivo/` | Diagnóstico de Apple, estrategia de cobertura, valoración, preguntas guía del caso, riesgos y recomendación final. |
| `2_dashboard/dashboard_ejecutivo.html` | Empresa, deuda y vencimientos, calificaciones, derivados, volatilidad, convertible, swaps, T-lock y futuros, escenarios y griegas. Usa las cifras que exporta el Rmd. |
| `3_reporte_reproducible/Proyecto_Final_Derivados.Rmd` | Reporte principal con el código visible. Genera el HTML y `resultados/resumen.json`. |
| `3_reporte_reproducible/datos/aapl_yahoo.csv` | Precios diarios de AAPL (Yahoo Finance, 8-nov-2021 a 6-nov-2026). |
| `3_reporte_reproducible/datos/griegas_equipo.csv` | Griegas de las 18 estrategias (LSEG, 12-oct-2026). También vienen incluidas dentro del Rmd. |
| `3_reporte_reproducible/resultados/resumen.json` | Cifras clave que exporta el Rmd y que usan el dashboard y el notebook de valuación. |
| `4_calculadora/calculadora_convertible.html` | Calculadora autocontenida con parámetros editables; reproduce las cifras del reporte. |
| `5_python/valuacion_bs_arbol_convertible.ipynb` y `.py` | Black-Scholes, árbol binomial, griegas, escenarios, convertible, validación con Molson Coors y call spread. |
| `5_python/garch_derivados_AAPL_Oro_WTI_6E.ipynb` y `.py` | Descarga de datos de Yahoo Finance, GARCH, correlaciones, VaR, Monte Carlo, árbol binomial y griegas para AAPL, oro, WTI, Euro FX y Nasdaq; su salida está en `resultados_garch.json`. |
| `6_uso_IA/` | Resumen de prompts, código recibido y correcciones (Claude y TecGPT) y la conversación completa con TecGPT con notas del equipo. |
| `7_anexo_capturas/` | PDF del anexo y las capturas originales por fuente: `lseg/`, `bloomberg/`, `cme/`, `tradingview/` e `investopedia/`. |
| `index.html` | Página de inicio del proyecto (GitHub Pages). |

## Cómo reproducir

**R Markdown**
1. Abrir `3_reporte_reproducible/Proyecto_Final_Derivados.Rmd` en RStudio.
2. Instalar los paquetes una vez:
   ```r
   install.packages(c("dplyr", "tidyr", "ggplot2", "knitr", "scales", "zoo", "jsonlite", "quantmod"))
   ```
3. Presionar **Knit**. El Rmd usa `datos/aapl_yahoo.csv`; si el archivo no existe, lo descarga de Yahoo Finance.

El GARCH(1,1) con distribución t se estima con código propio por máxima verosimilitud, así que no requiere `rugarch`;
si está instalado, el reporte lo usa para validar los parámetros.

**Python**: abrir cualquiera de los notebooks con las ligas de Colab de la tabla de entregables y ejecutar todo
(**Entorno de ejecución → Ejecutar todo**). El notebook de valuación lee los datos de este repositorio; el de GARCH los
descarga de Yahoo Finance. También pueden correrse localmente: `python valuacion_bs_arbol_convertible.py`.

## Fuentes y fechas

| Dato | Fecha | Fuente |
|---|---|---|
| Precios diarios de AAPL | 8-nov-2021 a 6-nov-2026 | Yahoo Finance |
| Treasuries | 6-nov-2026 | Forbes Advisor |
| Curvas SOFR y €STR | 2 y 3-nov-2026 | BlueGamma |
| EUR/USD | 6-nov-2026 | TradingNews |
| Operaciones | 30-sep a 6-nov-2026 | Investopedia |
| Spreads de AAPL, oro y WTI | 14-oct-2026 | Bloomberg OVME / OVML |
| 18 estrategias, griegas y curva SOFR | 12-oct-2026 | LSEG Workspace (`7_anexo_capturas/lseg/`) |
| Settle de oro y WTI, 6EZ6, FedWatch (ZQZ6), NQZ6 | 27 y 28-oct-2026 | CME Group |
| Deuda, efectivo y resultados de Apple | 26-sep-2026 | Apple, 10-K FY2026 |
| Derivados de Apple | 28-mar-2026 | Apple, 10-Q Q2 FY2026 |
| Emisión de Apple (US$7,500 M) | 5-nov-2026 | Suplemento de prospecto (SEC) |
| Calificaciones de Apple | Vigentes | Moody's (Aaa) y S&P (AA+) |
| Caso Molson Coors | Junio de 2007 | Notas de enseñanza NE90-06-107 |
| Diferenciales de Apple y basis EUR/USD | — | Supuestos del equipo |

## Operaciones: ejecutadas y modeladas

En Investopedia el equipo armó una posición protegida en AAPL: 100 acciones (30-sep), una put AAPL 330 como seguro
durante la primera semana de octubre (venció el 7-oct) y una call AAPL 345 con vencimiento el 18-dic para aprovechar el
alza. Además cotizó tres spreads en Bloomberg (AAPL, oro y WTI) y diseñó 18 estrategias con sus griegas en LSEG. Estas
últimas se reportan como modeladas: su resultado es teórico, no realizado.

## Seguridad

El proyecto no usa claves de API ni credenciales: Yahoo Finance se consulta sin clave. El `.gitignore` excluye archivos
de sesión de R y Jupyter (`.Rhistory`, `.RData`, `.Renviron`, `.ipynb_checkpoints`) y cualquier `.env`.

## Uso de inteligencia artificial

El código, el dashboard y la redacción se elaboraron con apoyo de Claude (Anthropic) y TecGPT, con datos, operaciones y
decisiones del equipo. El equipo revisó las cifras contra sus fuentes. El detalle está en
[`6_uso_IA/uso_de_IA_Equipo7.pdf`](6_uso_IA/uso_de_IA_Equipo7.pdf) y la conversación con TecGPT, en
[`6_uso_IA/conversacion_TecGPT_con_notas.pdf`](6_uso_IA/conversacion_TecGPT_con_notas.pdf).

*Material académico; no constituye recomendación de inversión.*
