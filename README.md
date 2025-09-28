Trabajo Práctico 2 – Inteligencia Artificial

Este repositorio contiene la implementación en Python de dos algoritmos de búsqueda aplicados a un problema simple en un eje unidimensional. El objetivo del trabajo es comparar una estrategia de búsqueda exhaustiva con una estrategia heurística, analizando sus características y resultados.

Algoritmos implementados:

Búsqueda Exhaustiva (BFS)
El algoritmo Breadth-First Search (BFS) recorre los estados por niveles utilizando una cola FIFO. Garantiza encontrar la solución más corta en cantidad de pasos siempre que todas las acciones tengan el mismo costo. Su implementación se encuentra en el archivo bfs.py. 

Para ejecutarlo: 
python3 bfs.py

El programa imprime las expansiones realizadas, el estado de la frontera y finalmente el camino reconstruido hasta la meta.

Búsqueda Heurística (A*)
El algoritmo A* utiliza una cola de prioridad ordenada por la función f(n) = g(n) + h(n), donde g(n) es el costo real acumulado y h(n) es la heurística de distancia absoluta al objetivo. Esta estrategia permite guiar la búsqueda de manera más eficiente que BFS, reduciendo el número de nodos explorados y garantizando una solución óptima en costo. Su implementación se encuentra en el archivo a-star.py. 

Para ejecutarlo: 
python3 a-star.py

Requisitos
	• Python 3.7 o superior
	• Probado en Python 3.12
	• No requiere librerías externas (solo collections, dataclasses, heapq, typing)

Evidencia

Ambos algoritmos imprimen información de depuración durante la ejecución (nodos expandidos, frontera y explorados) y al final presentan el camino encontrado, lo que permite comparar el comportamiento y eficiencia de cada enfoque.