# Para el proyecto de Servicio Social 

0. Imagenes

    * Se tienen videos en formato mp4, usar opencv para extraer frames a frame y asi sacar muchas imagenes de un video, despues aplicar cosas para hacerlas diferentes como rotaciones

    
1. Primero revisar las imagenes y aplicar normalizaciones 

    1.1 Que ya se uso antes 
        * Usar grabcut para el platillo segmentado
        *Redimensionado
        *Correcion de orientacion
        *Alineacion de imagenes

    1.2 Nuevas propuestas (Checar)

        Al final se tiene un platillo, pero el platillo generalmente se compone de 2 cosas

        1. El plato
        2. La comida

        Partiendo de eso, sabemos que los pixeles del plato se diferencian de los pixeles de la comida
        por lo que se propone lo siguiente para la normalizacion

        1. Entrenar un MLP para que sepa diferenciar entre pixeles de comida y pixeles de platillo

2. Extraccion de caracteristicas visuales

    2.1 Que ya se uso antes (probarlar)
        NUEVO
        *Probarlas no con la imagen completa si no con ventaneo

        la teoria es que 2 cosas
        1. Se tienen pocas imagenes
        2. Aun existe ruido y realmente no se puede pensar que 1 escalar describe a toda la imagen por igual

        por lo que se hara un ventaneo

        Tomando la imagen como una matriz de Rnxm 

        todas de la misma dimension, y tambien (en caso de que se realice el MLP) tendriamos una matriz de las mismas dimensiones con etiquetas de cada pixel

        el ventaneo se propondran rangos de ventanas para formar ventanas pequenas con pasos

        ej. 3x3

        se toma el pixel 1 (como no pixeles arriba a la izq se toma solo los adyacentes)

        es decir tomando el pixel como el centro de la ventana (y suma) quedaria

        | 1  |  a |   |
        | a  | a  |   |
        |    |    |   |   


        siendo la los adyacentes formando una matriz de 3x3 tomando como centro el pixel a analizar de tal forma que se tenga una matriz de caracteristicas por pixel con etiqueta doble (si es que se puede)

        pixe 1 X0, X1, X2 etc platillo, chef
        pixel final

        Se hara una redimension para rellenar los pixeles que falten porque si no, cada ventaneo hara mas pequenia la imagen

        1. Probar con el promedio
        2. o rellenando puros 0

3. Modelo a probar y producto principal

Se pretende formar un sistema difuso que pueda asignar calificaciones basandose en las caracteristicas

pero hacer reglas manuales con caracteristicas (no humanas) seria complejo

por lo que se busca probar con un genetic fuzzy o neural fuzzy para que se hagan varias cosas de manera automatica y no manual

1. Los rangos de las funciones
2. numero de conjuntos 
3. Tipo de funciones de cada conjunto

4. Reglas

de tal manera que el mismo algoritmo pueda dar reglas con las caracteristicas

