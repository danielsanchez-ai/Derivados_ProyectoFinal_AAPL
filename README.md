# Proyecto Final · Derivados y Administración de Riesgos

**Equipo 7 · Productos Derivados y Administración de Riesgos Corporativos · EGADE Business School**
Fecha de corte de los datos de mercado: **6 de noviembre de 2026**.

Análisis de Apple Inc. (AAPL) con volatilidad GARCH, simulación Monte Carlo, árboles binomiales, un bono convertible
(réplica del caso Molson Coors), swaps de divisas y de tasas, 18 estrategias de opciones sobre AAPL, oro y WTI, y las
operaciones realizadas en simuladores.

## Entrega

| Entregable | Liga |
|---|---|
| Reporte ejecutivo (PDF y Word) | [Abrir PDF](reporte_ejecutivo/Reporte_ejecutivo_Derivados.pdf) · [Word](reporte_ejecutivo/Reporte_ejecutivo_Derivados.docx) |
| Página del proyecto | [https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/](https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/) |
| Reporte completo (HTML) | [Abrir reporte](https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/Proyecto_Final_Derivados.html) |
| Dashboard ejecutivo | [Abrir dashboard](https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/dashboard/dashboard_ejecutivo.html) |
| Calculadora bono + opción y call spread | [Abrir calculadora](https://danielsanchez-ai.github.io/Derivados_ProyectoFinal_AAPL/calculadora/calculadora_convertible.html) |
| Presentación (PDF) | [Abrir presentación](presentacion/Presentacion_Proyecto_Final_Equipo7.pdf) |
| Uso de inteligencia artificial | [Abrir PDF](ia/uso_de_IA_Equipo7.pdf) |
| Código fuente | [`Proyecto_Final_Derivados.Rmd`](Proyecto_Final_Derivados.Rmd) |

## Contenido

| Ruta | Qué es |
|---|---|
| `Proyecto_Final_Derivados.Rmd` | Reporte principal en R Markdown. Genera el HTML y `resultados/resumen.json`. |
| `Proyecto_Final_Derivados.html` | Reporte compilado. |
| `dashboard/dashboard_ejecutivo.html` | Dashboard ejecutivo; lee las cifras de `resultados/resumen.json`. |
| `datos/aapl_yahoo.csv` | Precios diarios de AAPL (Yahoo Finance, 6-nov-2021 a 6-nov-2026). |
| `datos/griegas_equipo.csv` | Griegas de las 18 estrategias (LSEG, 12-oct-2026). También vienen incluidas dentro del Rmd. |
| `resultados/resumen.json` | Cifras clave que exporta el Rmd. |
| `notebooks/` | Notebook GARCH complementario (AAPL, oro, WTI, Euro FX y Nasdaq) y su salida. |
| `evidencias/` | Capturas de Investopedia y CME Group; pantallas y documento de estrategias en Bloomberg. |
| `calculadora/calculadora_convertible.html` | Calculadora interactiva del convertible (bono + opción) y del call spread, con parámetros editables. |
| `presentacion/` | Presentación en PDF (14 diapositivas). |
| `reporte_ejecutivo/` | Reporte ejecutivo en Word y PDF (4 páginas). |
| `ia/` | Prompts, código recibido y correcciones del uso de IA. |
| `index.html` | Página de inicio del proyecto (GitHub Pages). |

## Cómo reproducir

1. Abrir `Proyecto_Final_Derivados.Rmd` en RStudio con este repositorio como directorio de trabajo.
2. Instalar los paquetes una vez:
   ```r
   install.packages(c("dplyr", "tidyr", "ggplot2", "knitr", "scales", "zoo", "jsonlite", "quantmod"))
   ```
3. Presionar **Knit**. El Rmd usa `datos/aapl_yahoo.csv`; si el archivo no existe, lo descarga de Yahoo Finance.

El GARCH(1,1) con distribución t se estima con código propio por máxima verosimilitud, así que no requiere `rugarch`.
Si `rugarch` está instalado, el reporte lo usa para validar los parámetros.

## Fuentes y fechas

| Dato | Fecha | Fuente |
|---|---|---|
| Precios diarios de AAPL | 6-nov-2021 a 6-nov-2026 | Yahoo Finance |
| Treasuries | 6-nov-2026 | Forbes Advisor |
| Curvas SOFR y €STR | 2 y 3-nov-2026 | BlueGamma |
| EUR/USD | 6-nov-2026 | TradingNews |
| Operaciones | 30-sep a 6-nov-2026 | Investopedia |
| Spreads de AAPL, oro y WTI | 14-oct-2026 | Bloomberg OVME / OVML |
| 18 estrategias y griegas | 12-oct-2026 | LSEG |
| Settle de oro y WTI, 6EZ6, FedWatch, NQZ6 | 27 y 28-oct-2026 | CME Group |
| Deuda, efectivo y resultados de Apple | 26-sep-2026 | Apple, 10-K FY2026 |
| Derivados de Apple | 28-mar-2026 | Apple, 10-Q Q2 FY2026 |
| Emisión de Apple (US$7,500 M) | 5-nov-2026 | Suplemento de prospecto (SEC) |
| Calificaciones de Apple | Vigentes | Moody's (Aaa) y S&P (AA+) |
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

El código, el dashboard y la redacción se elaboraron con apoyo de Claude (Anthropic), con datos, operaciones y decisiones
del equipo. El equipo revisó las cifras contra sus fuentes. El detalle de prompts, código recibido y correcciones está en
[`ia/uso_de_IA_Equipo7.pdf`](ia/uso_de_IA_Equipo7.pdf).

*Material académico; no constituye recomendación de inversión.*
