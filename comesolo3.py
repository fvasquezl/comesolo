from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim

import random


# Definir la arquitectura de la red neuronal
class ComeSoloNet(nn.Module):
    def __init__(self):
        super(ComeSoloNet, self).__init__()
        self.fc1 = nn.Linear(15, 64)  # Entrada de 5x5 (tamaño del tablero)
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
    tablero_tensor = torch.FloatTensor(tablero).view(1, 15)

    # Cargar la red neuronal entrenada
    modelo = ComeSoloNet()
    modelo.load_state_dict(torch.load("modelo_entrenado.pth"))
    modelo.eval()  # Poner el modelo en modo de evaluación

    # Hacer la predicción
    with torch.no_grad():
        salida = modelo(tablero_tensor)
        mejor_movimiento = torch.argmax(salida).item()
        # Obtener el índice del movimiento más probable

    # Encuentra el origen y el destino del movimiento
    movimiento_mas_cercano = encontrar_movimiento_mas_cercano(tablero, mejor_movimiento)
    return movimiento_mas_cercano


def buscar_origen_destino(movimiento):
    # Encuentra el origen y el destino del movimiento
    for origen in range(1, 16):
        for destino in range(1, 16):
            if juego.movimiento_valido(origen, destino) and destino == movimiento:
                return origen, destino
    return 0, 0


def encontrar_movimiento_mas_cercano(tablero, mejor_movimiento):

    movimientos_validos = juego.obtener_movimientos_validos()
    distancia_minima = float("inf")
    movimiento_mas_cercano = (0, 0)

    for movimiento in movimientos_validos:
        distancia = abs(
            movimiento[1] - 1 - mejor_movimiento
        )  # Calcula la distancia entre el movimiento y la predicción
        if distancia < distancia_minima:
            distancia_minima = distancia
            movimiento_mas_cercano = movimiento

    if movimiento_mas_cercano == (0, 0):  # Si no se encuentra un movimiento válido
        print("No se encontraron movimientos válidos.")  # Mostrar un mensaje de error
        return (0, 0)  # Devolver un valor por defecto o lanzar una excepción

    return movimiento_mas_cercano


# Función para entrenar el modelo
def entrenar_modelo(
    num_datos=100000, num_epochs=100, lr=0.01, ruta_guardado="modelo_entrenado.pth"
):
    # Generar los datos de entrenamiento
    X_train = []
    y_train = []
    for _ in range(num_datos):
        juego = Comesolo()
        juego.ini_tablero()
        movimientos_validos = juego.obtener_movimientos_validos()
        if not movimientos_validos:
            continue
        movimiento_inicial = random.choice(movimientos_validos)
        juego.primer_movimiento(movimiento=movimiento_inicial[0])
        tablero_inicial = juego.tablero.copy()
        movimientos = []
        while True:
            movimientos_validos = juego.obtener_movimientos_validos()
            if not movimientos_validos:
                break
            origen, destino = random.choice(movimientos_validos)
            movimientos.append(origen)
            juego.realizar_movimiento(origen, destino)
        X_train.append(tablero_inicial)
        y_train.extend([movimiento - 1 for movimiento in movimientos])  # Corrección

    # Convertir los datos de entrenamiento a tensores
    # X_train = [torch.FloatTensor(x) for x in X_train]
    # y_train = torch.LongTensor(y_train)  # Ajustar índices de movimientos
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)

    # Instanciar la red, la función de pérdida y el optimizador
    modelo = ComeSoloNet()
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.SGD(modelo.parameters(), lr=lr)

    # Bucle de entrenamiento
    for epoch in range(num_epochs):
        for i, (X_batch, y_batch) in enumerate(zip(X_train, y_train)):
            # Hacer una pasada hacia adelante
            salidas = modelo(X_batch)
            y_batch = y_batch.unsqueeze(1)  # Ajustar la forma del tensor y_batch
            perdida = criterio(salidas, y_batch)

            # Retropropagación y optimización
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            # Imprimir la pérdida cada cierto número de épocas
            if epoch % 10 == 0 and i == 0:
                print(f"Época {epoch}, pérdida: {perdida.item()}")

    # Guardar los pesos entrenados
    torch.save(modelo.state_dict(), ruta_guardado)


class Comesolo:
    def __init__(self):
        self.tablero = self.ini_tablero()
        self.estado = 0

    def inc_estado(self):
        self.estado += 1

    def ini_tablero(self):
        self.tablero = [1] * 15

    def imprimir_tablero(self) -> None:
        print(f"Estado de tablero: {self.estado}")
        for i in range(5):
            inicio = i * (i + 1) // 2  # funcion para calcular numeros triangulares
            fin = inicio + i + 1
            fila = self.tablero[inicio:fin]
            fila_str = []
            for valor in fila:
                if valor == 0:
                    fila_str.append(f"\033[33m{valor}\033[0m")  # Color amarillo
                else:
                    fila_str.append(str(valor))
            print("  " * (4 - i) + "   ".join(fila_str))
        self.inc_estado()

    def primer_movimiento(self, movimiento: int) -> None:
        # Verifica que las posiciones sean válidas
        if not (1 <= movimiento <= 15):
            print("movimiento invalido")
            return
        else:
            self.tablero[movimiento - 1] = 0

    def movimiento_valido(self, origen: int, destino: int):
        # Verifica que las posiciones sean válidas
        if not (1 <= origen <= 15 and 1 <= destino <= 15):
            return False
        # Encuentra la posición intermedia entre origen y destino
        intermedia = (origen + destino) // 2

        # Verifica que el origen y la posición intermedia estén ocupados, y el destino esté vacío
        if (
            self.tablero[origen - 1] != 0
            and self.tablero[intermedia - 1] != 0
            and self.tablero[destino - 1] == 0
        ):
            # Verifica que el salto sea válido según las reglas del juego
            if abs(origen - destino) in [2, 3, 5, 7, 9]:  # Saltos
                return True
            long_nodes = [3, 4, 5, 12, 13, 14]
            if origen in long_nodes or destino in long_nodes:
                distance = [2, 3, 5, 7, 9]
            else:
                distance = [2, 3, 5, 7]
            1 = [3,5]
            2 = [5,7]
            3 = [5,7]
            4 =[3,2,7,9]
            5 =[7,9]
            6=[5,2,7,9]
            7=[5,2]
            8=[5,2]
            9=[7,2]
            10=[7,2]
            11=[7,2]
            12=[7,2]
            13=[2,2,7,9]
            14=[9,2]
            15=[9,2]


        return False

    def realizar_movimiento(self, origen, destino) -> None:
        print(origen, destino)
        if not self.movimiento_valido(origen, destino):
            print("Movimiento fuera de rango")
            return
        else:
            # calculamos la posicion intermedia entre origen y destino
            intermedia = (origen + destino) // 2
            # Realizamos el movimiento de eliminacion de fichas
            self.tablero[destino - 1] = self.tablero[origen - 1]
            self.tablero[origen - 1] = 0
            self.tablero[intermedia - 1] = 0

    # Modificar el método realizar_movimiento_ia
    def realizar_movimiento_ia(self):
        # tablero_actual = self.tablero.copy()  # Obtener una copia del tablero actual
        # mejor_movimiento = predecir_movimiento(tablero_actual)
        # print(mejor_movimiento)
        # self.realizar_movimiento(mejor_movimiento)
        origen, destino = predecir_movimiento(self.tablero)
        self.realizar_movimiento(origen, destino)

    def obtener_movimientos_validos(self):
        movimientos_validos = []
        for origen in range(1, 16):
            for destino in range(1, 16):
                if self.movimiento_valido(origen, destino):
                    movimientos_validos.append((origen, destino))
        return movimientos_validos

    def jugar(self):
        juego.ini_tablero()
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


if __name__ == "__main__":
    # entrenar el modelo
    entrenar_modelo()

    # Jugar el juego
    juego = Comesolo()
    juego.jugar()
