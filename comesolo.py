import torch
import torch.nn as nn
import torch.optim as optim


# Definición de la red neuronal
class RedNeuronalComesolo(nn.Module):
    def __init__(self, entrada, oculta, salida):
        super(RedNeuronalComesolo, self).__init__()
        self.fc1 = nn.Linear(entrada, oculta)
        self.fc2 = nn.Linear(oculta, salida)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Función para preprocesar el estado del tablero
def preprocesar_estado(estado_tablero):
    # Convertir el estado a un tensor
    tensor_estado = torch.tensor(estado_tablero, dtype=torch.float32)
    return tensor_estado


class Comesolo:
    def __init__(self):
        self.tablero = [0] * 15
        self.estado = 0
        # self.red_neuronal = RedNeuronalComesolo(
        #     15, 32, 120
        # )  # Ajustar los tamaños según sea necesario
        # self.red_neuronal.load_state_dict(
        #     torch.load("red_neuronal_comesolo.pth")
        # )  # Cargar los pesos entrenados

    def inc_estado(self):
        self.estado += 1

    def imprimir_tablero(self) -> None:
        for i in range(5):
            inicio = i * (i + 1) // 2
            fin = inicio + i + 1
            fila = self.tablero[inicio:fin]
            fila_str = []
            for valor in fila:
                if valor == 0:
                    fila_str.append(f"\033[33m{valor}\033[0m")  # Color amarillo
                else:
                    fila_str.append(str(valor))
            print(" " * (4 - i) + " ".join(fila_str))

        self.inc_estado()

    def iniciar_tablero(self) -> None:
        self.tablero = [1] * 15

    def primer_movimiento(self, movimiento: int) -> None:
        if not (1 <= movimiento <= 15):
            print("Posición inválida")
            return

        self.tablero[movimiento - 1] = 0

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

    def realizar_movimiento(self, movimiento) -> None:
        origen, destino = movimiento

        if not self.movimiento_valido(origen, destino):
            print("movimiento invalido")
            return
        self.tablero[destino - 1] = self.tablero[origen - 1]
        self.tablero[origen - 1] = 0
        self.tablero[((origen + destino) // 2) - 1] = 0

    def obtener_movimiento_optimo(self, estado_tablero):
        # Preprocesar el estado del tablero
        tensor_estado = preprocesar_estado(estado_tablero)

        # Obtener la predicción de la red neuronal
        with torch.no_grad():
            salida = self.red_neuronal(tensor_estado)
            prediccion = torch.argmax(salida)

        # Decodificar la predicción en las coordenadas de origen y destino
        origen = prediccion // 15 + 1
        destino = prediccion % 15 + 1

        return origen, destino

    # agregado por claude
    def jugar(self):
        self.imprimir_tablero()
        primer_movimiento = int(
            input(
                "Ingresa el índice de la ficha a eliminar para iniciar el juego (1-15): "
            )
        )
        self.primer_movimiento(movimiento=primer_movimiento)
        self.imprimir_tablero()

        while not self.verificar_fin_juego():
            origen_optimo, destino_optimo = self.obtener_movimiento_optimo(self.tablero)
            self.realizar_movimiento((origen_optimo, destino_optimo))
            self.imprimir_tablero()

        print("¡Has ganado!")

    def verificar_fin_juego(self):
        fichas_restantes = sum(1 for ficha in self.tablero if ficha != 0)
        return fichas_restantes == 1


if __name__ == "__main__":
    juego = Comesolo()
    juego.imprimir_tablero()
    juego.iniciar_tablero()
    juego.imprimir_tablero()
    juego.primer_movimiento(6)
    juego.imprimir_tablero()

    print(juego.tablero)
    # juego.jugar()


"""
Siguiendo con el codigo del juego comesolo, tengo hasta ahorita el siguiente codigo:  

class Comesolo:
    def __init__(self):
        self.tablero = [1] * 15
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
                print(f"\033[33m{self.tablero[i]:1d}  \033[0m ", end="")  # Amarillo
            else:
                print(f"{self.tablero[i]:1d}   ", end="")
        print(f"\nEstado de tablero: {self.estado}")
        self.inc_estado()

    def primer_movimiento(self, movimiento) -> None:
        self.tablero[movimiento - 1] = 0

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

    def realizar_movimiento(self, movimiento) -> None:
        origen, destino = movimiento

        if not self.movimiento_valido(origen, destino):
            print("movimiento invalido")
            return
        self.tablero[destino - 1] = self.tablero[origen - 1]
        self.tablero[origen - 1] = 0
        self.tablero[((origen + destino) // 2) - 1] = 0


if __name__ == "__main__":
    juego = Comesolo()
    juego.imprimir_tablero()
    primer_movimiento = int(input("Ingresa el primer movimiento: "))
    juego.primer_movimiento(movimiento=primer_movimiento)
    juego.imprimir_tablero()
    juego.realizar_movimiento(movimiento=(6, 13))
    juego.imprimir_tablero()

el cual contiene los siguientes metodos:
primer_movimiento(movimiento) el cual es el unico movimiento que realiza el usuario, por ejemplo eliminar la ficha en la
posicion 13
        0   
      0   1   
    1   1   1   
  1   1   1   1   
1   1   0   1   1 


de ahi la IA debe hacerse cargo utilizando el metodo realizar_movimiento(movimiento) que recibe
una tupla con 2 valores (origen, destino) que seran los movimientos seleccionados
que el metodo validar_movimento(origen, destino) validara si ese movimiento es valido
y se deben realizar todos lo movimientos necesarios que permita terminar el tablero
y dejarlo solo con un movimiento:
el cual puede ser algo como lo que se muestra acontinuacion, no es la solucion, pero es algo asi lo que 
se espera como solucion ideal.
         1
       2   3
     4   5   6
   7   8   9   10
 11  12   13  14  15

1,2,4,7,11



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
