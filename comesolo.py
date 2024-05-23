import tensorflow as tf
import numpy as np


class Comesolo:
    def __init__(self):
        self.tablero = [x for x in range(1, 16)]
        self.estado = 0

    def inc_estado(self):
        self.estado += 1

    def imprimir_tablero(self) -> None:
        fila = 1
        contador = 1
        print("  " * (5 - fila), end="")
        for i in range(len(self.tablero)):
            if i == contador:
                contador += fila + 1
                fila += 1
                print()
                print("  " * (5 - fila), end="")
            if self.tablero[i] == 0:
                print(f"\033[33m{self.tablero[i]:02d}  \033[0m", end="")  # Amarillo
            else:
                print(f"{self.tablero[i]:02d}  ", end="")
        print(f"\nEstado de tablero: {self.estado}")
        self.inc_estado()

    def primer_movimiento(self, movimiento) -> None:
        self.tablero[self.tablero.index(movimiento)] = 0

    def movimiento_valido(self, origen, destino):

        # Verifica que las posiciones sean válidas
        if not (1 <= origen <= 15 and 1 <= destino <= 15):
            return False

        # Encuentra la posición intermedia entre origen y destino
        # ejemplos:
        #  14 salta 12   14+12 = 26//2 = 13 (intermedia)
        #  05 salta 12   5+12 = 13//2 = 8 (intermedia)

        intermedia = (origen + destino) // 2
        if (
            self.tablero[origen - 1] != 0
            and self.tablero[intermedia - 1] != 0
            and self.tablero[destino - 1] == 0
        ):
            if abs(origen - destino) in [2, 3, 5, 7, 9]:  # Saltos
                return True
        return False

    def realizar_movimiento(self, movimiento):
        origen, destino = movimiento

        if not self.movimiento_valido(origen, destino):
            print("movimiento invalido")
            return
        self.tablero[destino - 1] = destino
        self.tablero[origen - 1] = 0
        self.tablero[((origen + destino) // 2) - 1] = 0


if __name__ == "__main__":
    juego = Comesolo()
    juego.imprimir_tablero()
    juego.primer_movimiento(11)
    juego.imprimir_tablero()
    juego.realizar_movimiento((6, 13))
    # juego.imprimir_tablero()
    # juego.realizar_movimiento((2, 9))
    # juego.imprimir_tablero()
    # juego.realizar_movimiento((13, 6))
    # juego.imprimir_tablero()
    # juego.realizar_movimiento((4, 13))
    # juego.imprimir_tablero()


"""
Tengo la idea de crear un programa en python 3.12 para resolver el juego llamado "comesolo" en espanol, o 
"Triangle Peg Game " en ingles, el programa debe ser resuelto usando red neuronal,debe utilizar programacion
orientada a objetos, y tratar de obtener siempre el resultado optimo.

- Explicacion del juego: 
El juego comesolo es un tablero con 15 posiciones dispuestas en forma triangular es decir:
         0
       0   0
     0   0   0
   0   0   0   0
 0   0   0   0   0
Esto reprenserntaria el tablero vacio.

Para iniciar el juego deben colocarse todas las fichas en el tablero, para lo cual yo voy a poner el indice de cada posicion,
para mejor explicacion, pero el tablero puede rellenarse con 1 y quedaria de la siguiente manera
        1
       2   3
     4   0   6
   7   8   9   10
 11  12  13  14  15
Esto representaria el estado inicial del tablero con las fichas colocadas.

- Inicio del juego: 
El juego se inicia quitando una ficha cualquiera, esto lo hace el jugador, por ejemplo quitemos la ficha 5:
         1
       2   3
     4   0   6
   7   8   9   10
 11  12  13  14  15

- Reglas del juego: 
Una vez iniciado el juego, para eliminar la siguiente ficha, debe existir dos fichas contiguas al espacio en blanco en 
linea directa, se toma la segunda ficha, saltando la contigua al espacio en blanco y aterrizando en el espacio vacio,de esta
manera la ficha saltada es eliminada.

Continuando con el ejemplo se tendrian las siguientes posibilidades:

1. Eliminar la ficha 9 saltando desde la posicion 15 a la 5
         1
       2   3
     4   0   6
   7   8   9   10
 11  12  13  14  15

2. o tambien Eliminar la ficha 8 saltando desde la posicion 12 a la 5
         1
       2   3
     4   0   6
   7   8   9   10
 11  12  13  14  15

Se debe elegir entre una de las opciones anteriores, seleccionemos la numero 1, el resultado seria el siguiente:

         1
       2   3
     4   5   6
   7   8   0   10
 11  12  13  0  15

Ahora las posiblidades de movimiento son: 
1. Eliminar el 5 saltando de la posicion 2 a la 9
2. Eliminar el 13 saltando de la posicion 12 a la 14
3. Eliminar el 8 saltando de la posicion 7 a la 10

Y asi se debe continuar hasta que el tablero quede con una sola ficha.

A partir del primer movimiento la red neuronal debe hacerse cargo de reslover el juego entregando la solucion ideal
y cada moviemto seleccionado.
"""
