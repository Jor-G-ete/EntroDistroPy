

# Algoritmos de bayes seleccionados

Debido a los resultados tras usar el dataset del dia: ***datos_discretizar_04052019.csv***, se valor y se decide quedarse con los modelos **Multinomial bayesiano** y **Complement bayesiano** , los cuales son 1 y 2 en el conf.yalm

![](./images/4.png)



# Valoración/Calificación

Proceso por el cual se valora el **ROI**, para ello lo que haremos será dividir el ROI , entre positivo, negativo y neutro. 

Tomando los datos positivos, estos los dividiremos en 10 grupos:

```
muy alto
alto
alto moderado
medio alto
medio
medio moderado
bajo alto
bajo
bajo moderado
```

Tras ello nos haremos una pregunta:

*¿Sube el ROI?*

Con la respuesta:

```
respuesta = [SI] sera el conjunto cerrado ( alto_moderado, medio_alto ,medio,medio_moderado)

respuesta = [NO] el conjunto cerrado (bajo_alto,bajo,bajo_moderado,deficiente,neutro)
```

Todo esto se podrá visualizar como dos columnas extra, que a la hora de discretizar no se tendrán en cuenta por el ruido y los  problemas que dan al hacer las operaciones:

![](./Images/1.png)





Debido  a que se añaden las categorías neutro y negativo, estas tendrán un valor en la discretización de -1 y -2 respectivamente. Por ello si se activa la opción de normalizar se aumentara en 2 todas las categorías para que así, quede que la categoría negativa es 0 y la neutra es 1.



# Discretización

## Entrópicamente

.....



## x bins

![2](./Images/2.png)

A primera vista puede parecer que la discretización para 10 bins ha sido errores, puesto que el rango de valores va desde 0 hasta 50. Pero todo lo contrario. 

Discretizada cada columna para 10 bins, solo que según se mueve a nuevas columnas ( hacia la derecha), para evitar repetir valores, le añade 10. Siguiendo esta formula:
$$
(Numero\ de\ columna * 10) + valor 
$$
De esta manera llegamos a:

* Columna 0 = 0 *10 + valor = valor
* Columna 1 = 1*10 + valor = valor +10
* Columna 2 = 2*10 + valor = valor +20
* ...
* Columna 5 = 5*10 + valor = valor +50



La salida se modifica para que este entre 0 y el numero de bins , pero esta característica se puede desactivar. Poniendo el parámetro (Normaliza ) de la función DiscretizaBins a False.

![](./Images/3.png)