class Comesolo:
    def __init__(self):
        self.tablero = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
        self.estado = 0
        self.iniciar_tablero()

    def iniciar_tablero(self):
        contador = 1
        self.estado += 1
        for i in range(len(self.tablero)):
            for j in range(len(self.tablero[i])):
                self.tablero[i][j] = contador
                contador += 1
        # for fila in self.tablero:
        #     for elemento in fila:
        #         elemento = elemento.index+1* len(fila)

    def imprimir_tablero(self):
        print(f"Estado de tablero: {self.estado}")
        max_positions = len(self.tablero[-1])

        for fila in self.tablero:
            espacios_en_blanco = "  " * (max_positions - len(fila))
            valores_alineados = [str(valor).zfill(2) for valor in fila]
            print(espacios_en_blanco + "  ".join(valores_alineados))

    def iniciar_juego(self):
        while True:
            try:
                movimiento = int(
                    input(
                        "Primera ficha a eliminar para iniciar el juego[1-15], salir[0]:"
                    )
                )
                if movimiento == 0:
                    break
                elif movimiento >= 1 and movimiento <= 15:
                    if self.primer_movimiento(movimiento):
                        break
                else:
                    print("Valor fuera de rango")
            except ValueError:
                print("Error en el ingreso de datos")

    def primer_movimiento(self, movimiento):
        for fila in self.tablero:
            if movimiento in fila:
                fila[fila.index(movimiento)] = 0
                self.estado += 1
                return True
        return False

    def verificar_fin_juego(self):
        contador_fichas = sum(fila.count(0) for fila in self.tablero)
        if contador_fichas == len(self.tablero) * (len(self.tablero) + 1) // 2 - 1:
            print("¡Has ganado!")
            return True
        return False


if __name__ == "__main__":
    tablero = Comesolo()
    tablero.imprimir_tablero()
    tablero.iniciar_juego()
    tablero.imprimir_tablero()
    if not tablero.verificar_fin_juego():
        print("El juego continúa...")


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
