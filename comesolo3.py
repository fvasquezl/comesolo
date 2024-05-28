from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim

import random


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
    tablero_tensor = torch.FloatTensor(tablero).view(1, -1)

    # Cargar la red neuronal entrenada
    modelo = ComeSoloNet()
    modelo.load_state_dict(torch.load("modelo_entrenado.pth"))
    modelo.eval()  # Poner el modelo en modo de evaluación

    # Hacer la predicción
    with torch.no_grad():
        salida = modelo(tablero_tensor)
        _, prediccion = torch.max(salida, 1)
        mejor_movimiento = prediccion.item() + 1  # Ajustar el índice de movimiento

    return mejor_movimiento


# Función para entrenar el modelo
def entrenar_modelo(
    num_datos=10000, num_epochs=100, lr=0.01, ruta_guardado="modelo_entrenado.pth"
):
    # Generar los datos de entrenamiento
    X_train = []
    y_train = []
    for _ in range(num_datos):
        juego = Comesolo()
        juego.ini_tablero()
        tablero_inicial = juego.tablero.copy()
        movimientos = []
        while True:
            movimientos_validos = juego.obtener_movimientos_validos()
            if not movimientos_validos:
                break
            origen, destino = random.choice(movimientos_validos)
            movimientos.append((origen, destino))
            juego.realizar_movimiento(origen, destino)
        X_train.append(tablero_inicial)
        y_train.extend(movimientos)

    # Convertir los datos de entrenamiento a tensores
    X_train = [torch.FloatTensor(x) for x in X_train]
    y_train = torch.LongTensor(
        [movimiento - 1 for movimiento in y_train]
    )  # Ajustar índices de movimientos

    # Instanciar la red, la función de pérdida y el optimizador
    modelo = ComeSoloNet()
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.SGD(modelo.parameters(), lr=lr)

    # Bucle de entrenamiento
    for epoch in range(num_epochs):
        for X_batch, y_batch in zip(X_train, y_train):
            # Hacer una pasada hacia adelante
            salidas = modelo(X_batch)
            perdida = criterio(salidas, y_batch)

            # Retropropagación y optimización
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            # Imprimir la pérdida cada cierto número de épocas
            if epoch % 10 == 0:
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

        return False

    def realizar_movimiento(self, origen, destino) -> None:
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
        tablero_actual = self.tablero.copy()  # Obtener una copia del tablero actual
        mejor_movimiento = predecir_movimiento(tablero_actual)
        self.realizar_movimiento(mejor_movimiento)

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
