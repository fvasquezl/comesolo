from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim

import csv

import random


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
def predecir_movimiento(tablero, epsilon=0.1):
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

    # # Exploración Epsilon-Greedy
    # if random.random() < epsilon:
    #     # Exploración: seleccionar un movimiento aleatorio
    #     mejor_movimiento = random.randint(0, 14)
    # else:
    #     # Explotación: seleccionar el mejor movimiento predicho
    #     mejor_movimiento = mejor_movimiento_predicho

    # Encuentra el origen y el destino del movimiento
    origen, destino = buscar_origen_destino(mejor_movimiento)
    tablero_tensor = juego.tablero

    return origen, destino


def buscar_origen_destino(movimiento):
    # Encuentra el origen y el destino del movimiento
    for origen in range(1, 16):
        for destino in range(1, 16):
            if juego.movimiento_valido(destino, origen) and destino == movimiento:
                return origen, destino
    return 0, 0


def entrenar_modelo(
    num_datos=10000, num_epochs=100, lr=0.01, ruta_guardado="modelo_entrenado.pth"
):
    # Generar los datos de entrenamiento
    X_train = []
    y_train = []
    movimiento_inicial = 0
    for _ in range(num_datos):
        juego = Comesolo()
        juego.ini_tablero()
        if movimiento_inicial > 15:
            movimiento_inicial = 0
        movimiento_inicial += 1
        juego.primer_movimiento(movimiento_inicial)
        tablero_inicial = juego.tablero.copy()
        # juego.imprimir_tablero()
        movimientos = []
        while True:
            movimientos_validos = juego.obtener_movimientos_validos()
            if not movimientos_validos:
                break
            origen, destino = random.choice(movimientos_validos)
            movimientos.append(origen)
            juego.realizar_movimiento(origen, destino)

        if sum(juego.tablero) != 1:
            continue
        if movimientos in y_train:
            continue

        print(tablero_inicial)
        juego.imprimir_tablero()
        print(movimientos)
        X_train.append(tablero_inicial)

        y_train.extend([movimiento for movimiento in movimientos])  # Corrección
        # print(X_train)
        # print(y_train)

    # with open("tablero.txt", "w") as f:
    #     csv.writer(f, delimiter=",").writerows(X_train)

    # with open("movimiento.txt", "w") as f:
    #     csv.writer(f, delimiter=",").writerows(y_train)

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

    def movimiento_valido(self, destino: int, origen: int):
        # Verifica que las posiciones sean válidas
        if not (1 <= destino <= 15 and 1 <= origen <= 15):
            return False
        # Encuentra la posición intermedia entre destino y origen
        intermedia = (destino + origen) // 2

        # Verifica que el destino y la posición intermedia estén ocupados, y el origen esté vacío
        if (
            self.tablero[destino - 1] == 0
            and self.tablero[intermedia - 1] != 0
            and self.tablero[origen - 1] != 0
        ):
            reglas = {
                1: [4, 6],
                2: [7, 9],
                3: [8, 10],
                4: [1, 6, 11, 13],
                5: [12, 14],
                6: [1, 4.13, 15],
                7: [2, 9],
                8: [3, 10],
                9: [2, 7],
                10: [3, 8],
                11: [4, 13],
                12: [5, 14],
                13: [4, 6, 11, 15],
                14: [5, 12],
                15: [6, 13],
            }

            if destino in reglas.keys():
                if origen in reglas[destino]:
                    return True
        return False

    def realizar_movimiento(self, origen, destino) -> None:
        # print(origen, destino)
        if not self.movimiento_valido(destino, origen):
            print("Movimiento fuera de rango")
            return
        else:
            # calculamos la posicion intermedia entre origen y destino
            intermedia = (origen + destino) // 2
            # Realizamos el movimiento de eliminacion de fichas
            self.tablero[destino - 1] = self.tablero[origen - 1]
            self.tablero[origen - 1] = 0
            self.tablero[intermedia - 1] = 0

    def obtener_movimientos_validos(self):
        movimientos_validos = []
        for destino in range(1, 16):
            for origen in range(1, 16):
                if self.movimiento_valido(destino, origen):
                    movimientos_validos.append((origen, destino))
        return movimientos_validos

    def realizar_movimiento_ia(self):
        tablero_actual = self.tablero.copy()  # Obtener una copia del tablero actual
        # mejor_movimiento = predecir_movimiento(tablero_actual)
        # print(mejor_movimiento)
        # self.realizar_movimiento(mejor_movimiento)
        origen, destino = predecir_movimiento(tablero_actual)
        self.realizar_movimiento(origen, destino)

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
    entrenar_modelo()

    # Jugar el juego
    juego = Comesolo()
    juego.jugar()
