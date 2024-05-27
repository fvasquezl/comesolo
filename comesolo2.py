from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim


# Definir la arquitectura de la red neuronal
class ComeSoloNet(nn.Module):
    def __init__(self):
        super(ComeSoloNet, self).__init__()
        self.fc1 = nn.Linear(25, 64)  # Entrada de 5x5 (tamaño del tablero)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 15)  # Salida de 15 movimientos posibles

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Función para predecir el mejor movimiento
def predecir_movimiento(tablero):
    # Preprocesar el tablero a formato de tensor
    print(tablero)
    tablero_tensor = torch.FloatTensor(tablero).view(1, -1)

    # Cargar la red neuronal entrenada
    modelo = ComeSoloNet()
    modelo.load_state_dict(torch.load("modelo_entrenado.pth"))

    # Hacer la predicción
    with torch.no_grad():
        salida = modelo(tablero_tensor)
        _, prediccion = torch.max(salida, 1)
        mejor_movimiento = prediccion.item() + 1  # Ajustar el índice de movimiento

    return mejor_movimiento


class Comesolo:
    def __init__(self):
        self.tablero = self.ini_tablero()
        self.estado = 0

    def inc_estado(self):
        self.estado += 1

    def ini_tablero(self):
        self.tablero = [[1] * x for x in range(1, 6)]

    def imprimir_tablero(self) -> None:
        print(f"Estado de tablero: {self.estado}")
        for fila in self.tablero:
            fila_str = []
            for valor in fila:
                if valor == 0:
                    fila_str.append(colored(str(valor), "yellow"))
                else:
                    fila_str.append(str(valor))
            print("  " * (5 - len(fila)) + "   ".join(fila_str))
        self.inc_estado()

    def primer_movimiento(self, movimiento: int) -> None:
        # Verifica que las posiciones sean válidas
        if not (1 <= movimiento <= 15):
            print("movimiento invalido")
            return False
        else:
            c_origen, c_destino = self.obtener_coordenadas(movimiento)
            self.tablero[c_origen][c_destino] = 0

    def movimiento_valido(self, origen: int, destino: int):
        # Verifica que las posiciones sean válidas
        if not (1 <= origen <= 15 and 1 <= destino <= 15):
            return False

        fila_origen, col_origen = self.obtener_coordenadas(origen)
        fila_destino, col_destino = self.obtener_coordenadas(destino)

        # Encuentra la posición intermedia entre origen y destino
        fila_intermedia = (fila_origen + fila_destino) // 2
        col_intermedia = (col_origen + col_destino) // 2

        # Verifica que el origen y la posición intermedia estén ocupados, y el destino esté vacío
        if (
            self.tablero[fila_origen][col_origen] != 0
            and self.tablero[fila_intermedia][col_intermedia] != 0
            and self.tablero[fila_destino][col_destino] == 0
        ):
            # Verifica que el salto sea válido según las reglas del juego
            if (
                abs(fila_origen - fila_destino) == 2
                or abs(col_origen - col_destino) == 2
            ):
                return True

        return False

    def obtener_coordenadas(self, indice: int):
        if not (1 <= indice <= sum(range(1, len(self.tablero) + 1))):
            raise ValueError("Índice fuera de rango")

        for fila in range(len(self.tablero)):
            inicio = sum(range(fila + 1))
            fin = inicio + len(self.tablero[fila])
            if inicio < indice <= fin:
                col = indice - inicio
                return fila, col - 1

    def realizar_movimiento(self, origen, destino) -> None:
        if self.movimiento_valido(origen, destino):
            fila_origen, col_origen = self.obtener_coordenadas(origen)
            fila_destino, col_destino = self.obtener_coordenadas(destino)
            fila_intermedia = (fila_origen + fila_destino) // 2
            col_intermedia = (col_origen + col_destino) // 2

            self.tablero[fila_destino][col_destino] = self.tablero[fila_origen][
                col_origen
            ]
            self.tablero[fila_origen][col_origen] = 0
            self.tablero[fila_intermedia][col_intermedia] = 0
        else:
            print("Movimiento inválido")

    # Modificar el método realizar_movimiento_ia
    def realizar_movimiento_ia(self):
        tablero_actual = self.tablero.copy()  # Obtener una copia del tablero actual
        mejor_movimiento = predecir_movimiento(tablero_actual)
        self.realizar_movimiento(mejor_movimiento)


if __name__ == "__main__":
    juego = Comesolo()
    juego.ini_tablero()
    juego.imprimir_tablero()
    movimiento_inicial = int(input("Ingrese su movimiento inicial (1-15): "))
    juego.primer_movimiento(movimiento_inicial)
    juego.imprimir_tablero()

    while True:
        juego.realizar_movimiento_ia()
        juego.imprimir_tablero()
        movimientos_validos = juego.obtener_movimientos_validos()
        if not movimientos_validos:
            break

    print("Juego terminado.")

"""
Este juego  es muy sencillo, se inicia con un movimiento, que lo hace el usuario, en este caso
 ejecutando el metodo primer_movimiento(movimiento), lo cual remueve una ficha del tablero, 
 y apartir de ahi la IA debe hacerse cargo, moviendo una ficha, saltando una ficha intermedia 
 y aterrizando en la posicion vacia, volver a repetir el movimento hasta que el tablero quede con 
 una sola ficha que es la solucion ideal 
"""
