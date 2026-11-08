# Guion del screencast (≈10 minutos)

Cada bloque indica el tiempo, qué mostrar en pantalla y qué decir. El texto es una guía: díganlo con sus palabras. Las mismas notas están en la presentación (notas del orador).

**Antes de grabar:** cierren pestañas con correos o datos personales; en Investopedia muestren solo Holdings y Trade History; tengan abiertos la presentación, el dashboard y el repositorio.

## 1. 0:00–0:30

**En pantalla:** Diapositiva 1 (portada).

**Guion:** Buen día. Somos el Equipo 7. Presentamos nuestro proyecto final: cómo Apple puede medir y administrar sus riesgos de mercado con derivados. Todos los datos de mercado están al cierre del 6 de noviembre de 2026, para que las cifras sean comparables entre secciones.

## 2. 0:30–1:00

**En pantalla:** Diapositiva 2.

**Guion:** El proyecto tiene cuatro bloques. Primero medimos la volatilidad de Apple con un GARCH y la usamos para valuar la call que compramos. Después replicamos el caso Molson Coors con un convertible de Apple. Luego cubrimos una emisión en euros con un cross-currency swap y un swap de tasas. Al final revisamos las 18 estrategias con sus griegas y las operaciones que sí ejecutamos en Investopedia.

## 3. 1:00–2:00

**En pantalla:** Diapositiva 3, luego el dashboard en la sección Volatilidad (gráfica de volatilidad condicional).

**Guion:** Estimamos un GARCH(1,1) con errores t de Student sobre cinco años de rendimientos diarios. La persistencia es 0.978: los periodos de alta volatilidad duran, un shock tarda unos 32 días en perder la mitad de su efecto. Hoy la volatilidad condicional es 29.3% y la de largo plazo 30.3%, así que el modelo espera que suba un poco. Los 4.5 grados de libertad confirman colas pesadas. La versión anterior usaba dos años y la volatilidad de un solo día; ahora usamos el promedio esperado hasta el vencimiento de la opción, 29.5%.

## 4. 2:00–3:00

**En pantalla:** Diapositiva 4, luego en el dashboard la gráfica “Valor de la call según la volatilidad”.

**Guion:** Valuamos la call 345 que compramos el 27 de octubre con un árbol binomial americano. Con la volatilidad GARCH vale 13.18 dólares; el mercado la paga en 11.25, lo que implica una volatilidad de 25.4%. Es decir, GARCH espera más movimiento del que cobra el mercado y la opción se ve barata. Antes comparábamos con la volatilidad de un día, 33.6%, lo que sobreestimaba el valor. Ojo: esto es una señal, no una ganancia asegurada.

## 5. 3:00–3:45

**En pantalla:** Diapositiva 5, luego el abanico de Monte Carlo en el dashboard.

**Guion:** Para medir el riesgo de la posición simulamos 10,000 trayectorias hasta el vencimiento, cada una con su propia volatilidad GARCH. En 90% de los casos Apple termina entre 296 y 406 dólares. La call termina dentro del dinero en 53% de los escenarios, pero solo en 36% el precio supera 358.50, que es lo necesario para recuperar la prima. Estas probabilidades son del mundo real, por eso no se usan para valuar; para eso está el árbol.

## 6. 3:45–5:00

**En pantalla:** Diapositiva 6, luego la tabla “Valor presente” del convertible en el dashboard.

**Guion:** Replicamos el caso Molson Coors con una emisión hipotética de Apple: 10,000 millones a seis años con 30% de prima de conversión. El cupón baja de 5.09% a 0.82%, unos 426 millones menos de intereses al año. Pero el valor presente de esa reducción, 2,181.7 millones, es exactamente el valor de las opciones que Apple regala a los inversionistas. No es coincidencia: el cupón se calcula para que bono más opción valgan la par. Por eso el VPN es cero y eliminamos la "diferencia positiva" de 1,269.90 millones de la versión anterior. El call spread cuesta 736.6 millones, 7.4% del principal, y solo traslada el costo de evitar dilución.

## 7. 5:00–6:00

**En pantalla:** Diapositiva 7, luego la gráfica de escenarios de EUR/USD en el dashboard.

**Guion:** Apple emite 5,000 millones de euros a siete años y los convierte a dólares con un cross-currency swap. El costo resultante es 5.02% en dólares, 12 puntos base menos que emitir directo en dólares. La versión anterior decía que el swap era más caro, pero comparaba con un tipo de cambio fijo; el forward a siete años es 1.2192, porque el mercado ya descuenta el diferencial de tasas. Lo importante es la cobertura: si el euro se aprecia 20%, la deuda cuesta 1,119 millones más, pero el swap gana 1,152; el neto apenas se mueve.

## 8. 6:00–6:45

**En pantalla:** Diapositiva 8.

**Guion:** Después convertimos ese 5.02% fijo a flotante con un swap de tasas. Aquí estaba un error importante: antes recibíamos 5.02%, la misma tasa que pagábamos. La tasa de mercado del swap a siete años es 4.44%, así que el costo correcto es SOFR más 58 puntos base. El swap vale cero al inicio; si SOFR baja 100 puntos gana 355 millones, si sube 100 pierde 332. No elimina el riesgo de tasas: cambia riesgo de valor por riesgo de flujo.

## 9. 6:45–7:30

**En pantalla:** Diapositiva 9, luego la gráfica de Delta y Vega en el dashboard.

**Guion:** Armamos 18 estrategias, seis por subyacente, y sumamos sus griegas. Los straddles concentran gamma y vega pero pierden mucho por tiempo; los short iron butterflies son lo opuesto. El portafolio de WTI queda casi neutral en delta, pero no en vega. Las cifras de oro son más grandes por el tamaño del contrato. El mensaje: combinar largos y cortos cambia el perfil de riesgo, no lo elimina.

## 10. 7:30–8:30

**En pantalla:** Diapositiva 10, luego la captura de Holdings (evidencias/investopedia) y la tabla de operaciones del dashboard.

**Guion:** En el simulador ejecutamos solo tres operaciones: 100 acciones de Apple, una put 330 como seguro, que venció sin valor, y la call 345 de diciembre. Al cierre del 6 de noviembre la cuenta vale 1,000,613 dólares: 1,157 de ganancia no realizada en acciones, menos 225 en la call y menos 320 realizados en la put. Separamos lo ejecutado de lo modelado: los spreads de oro y WTI cotizados en Bloomberg tienen un resultado teórico negativo, pero no son pérdidas realizadas. Aquí podemos mostrar la pantalla de Holdings.

## 11. 8:30–9:15

**En pantalla:** Diapositiva 11, y 5 segundos del Rmd abierto en RStudio para mostrar el código.

**Guion:** Atendimos los comentarios de la revisión. Usamos una sola fecha de corte; el convertible ya no muestra un ahorro que no existe; el CCS se compara con el forward; el IRS recibe la tasa de mercado; el árbol usa la volatilidad promedio de GARCH y se compara con el precio real; el inventario distingue lo ejecutado de lo modelado; y todo es reproducible desde el repositorio, con el dashboard leyendo las mismas cifras que el reporte.

## 12. 9:15–10:00

**En pantalla:** Diapositiva 12; al final, el repositorio de GitHub en el navegador.

**Guion:** Cerramos con cuatro ideas. Una opción se valúa con la volatilidad esperada durante su vida. La diferencia entre modelo y mercado es una señal, no una ganancia. Un cupón menor en un convertible no es un ahorro: se paga con opciones. Y los swaps valen cero al inicio; su valor está en la cobertura. En resumen, los derivados no crean valor por sí solos, trasladan riesgo a un precio. Todo el código, los datos y el dashboard están en nuestro repositorio de GitHub. Gracias.
