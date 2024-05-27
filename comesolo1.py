import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# Definición de la red neuronal
class RedNeuronalComesolo(nn.Module):
    def __init__(self, entrada, oculta, salida):
        super(RedNeuronalComesolo, self).__init__()
        self.fc1 = nn.Linear(entrada, oculta)
        self.fc2 = nn.Linear(oculta, 15)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(
            self.fc2(x), dim=1
        )  # Aplica softmax para obtener probabilidades
        return x


# Función para preprocesar el estado del tablero
def preprocesar_estado(estado_tablero):
    # Convertir el estado a un tensor
    tensor_estado = torch.tensor(estado_tablero, dtype=torch.float32)
    return tensor_estado


class Comesolo:
    def __init__(self):
        self.tablero = [1] * 15
        self.estado = 0
        self.red_neuronal = RedNeuronalComesolo(
            15, 32, 120
        )  # Ajustar los tamaños según sea necesario
        self.red_neuronal.load_state_dict(
            torch.load("red_neuronal_comesolo.pth"), strict=False
        )

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

    def obtener_movimiento_optimo(self, estado_tablero):
        # Preprocesar el estado del tablero
        print(estado_tablero)
        tensor_estado = preprocesar_estado(estado_tablero)
        print(tensor_estado)

        # Obtener la predicción de la red neuronal
        with torch.no_grad():
            salida = self.red_neuronal(tensor_estado)
            print("salida", salida)
            prediccion = torch.argmax(salida)
            print("prediccion", prediccion)

        # Decodificar la predicción en las coordenadas de origen y destino
        origen = prediccion // 15 + 1
        destino = prediccion % 15 + 1

        return origen, destino

        # Función de recompensa

    def recompensa(self, tablero, fichas_IA):
        """Devuelve la recompensa para el agente."""
        if tablero.count(0) == 0:  # Si el agente ganó
            return 10  # Alta recompensa por ganar
        elif tablero.count(0) == 1:  # Si el agente perdió
            return -5  # Alta penalización por perder
        else:
            return fichas_IA  # Recompensa por cada ficha que la IA elimine

    def estado_a_vector(self):
        """Convierte el tablero del juego a un vector numérico."""
        vector = np.zeros(15)
        for i, valor in enumerate(self.tablero):
            if valor == 0:
                vector[i] = 1  # Marca la posición vacía con 1
            else:
                vector[i] = 0  # Marca las posiciones con fichas con 0
        return vector

    def movimientos_posibles(self):
        """Devuelve una lista de movimientos posibles en el estado actual."""
        movimientos = []
        for origen in range(1, 16):
            for destino in range(1, 16):
                if self.movimiento_valido(origen, destino):
                    movimientos.append((origen, destino))
        return movimientos

    # Función para entrenar el modelo
    def entrenar(self, num_iteraciones, tasa_aprendizaje=0.001):
        """Entrena la red neuronal para resolver el juego."""
        optimizador = optim.Adam(self.red_neuronal.parameters(), lr=tasa_aprendizaje)
        criterio = nn.CrossEntropyLoss()

        for i in range(num_iteraciones):
            estado = self.estado_a_vector()

            # Movimiento inicial del jugador (simulando al jugador)
            movimiento_humano = random.randint(1, 15)
            self.primer_movimiento(movimiento=movimiento_humano)
            estado = self.estado_a_vector()
            fichas_IA = self.tablero.count(0)

            while True:
                # Obtener la predicción de la red neuronal
                # probabilidades = self.red_neuronal(np.expand_dims(estado, axis=0))
                probabilidades = self.red_neuronal(
                    torch.tensor(estado, dtype=torch.float32)
                )

                movimientos_posibles = self.movimientos_posibles()

                epsilon = 0.1
                if random.random() < epsilon:
                    # Elige un movimiento aleatorio válido
                    if movimientos_posibles:  # Verifica si hay movimientos disponibles
                        accion = random.choice(movimientos_posibles)
                    else:
                        # No hay movimientos posibles, la IA no puede jugar
                        break
                else:
                    # Seleccionar una acción (con la mayor probabilidad)
                    accion = np.argmax(probabilidades.detach().numpy())

                # Encuentra el movimiento válido
                # movimientos = self.movimientos_posibles()
                if movimientos_posibles:
                    movimiento_elegido = movimientos_posibles[
                        accion
                    ]  # Obtiene el movimiento de la lista
                    origen, destino = movimiento_elegido  # Desempaqueta la tupla
                    self.realizar_movimiento(movimiento_elegido)
                    break

                # Calcular la recompensa
                recompensa_actual = self.recompensa(self.tablero, fichas_IA)

                self.red_neuronal.zero_grad()
                prediccion = self.red_neuronal(
                    torch.tensor(estado, dtype=torch.float32)
                )
                loss = criterio(prediccion, torch.tensor([accion], dtype=torch.float))
                loss.backward()
                optimizador.step()

                # Si la partida termina, sale del bucle
                if recompensa_actual != 0:
                    break

                # Actualiza el estado del juego
                estado = self.estado_a_vector()
                fichas_IA = self.tablero.count(0)

            # Imprime el progreso del entrenamiento (opcional)
            print(f"Iteración {i+1}")

        # Guardar los pesos entrenados
        torch.save(self.red_neuronal.state_dict(), "red_neuronal_comesolo.pth")

    # agregado por claude
    def jugar(self):
        while True:
            self.tablero = [1] * 15
            self.imprimir_tablero()
            primer_movimiento = int(
                input(
                    "Ingresa el índice de la ficha a eliminar para iniciar el juego (1-15): "
                )
            )
            self.primer_movimiento(movimiento=primer_movimiento)
            self.imprimir_tablero()

            # # Carga la red neuronal desde el archivo .pth
            # self.red_neuronal = RedNeuronalComesolo(15, 32, 120)  # Crea la red neuronal
            # self.red_neuronal.load_state_dict(
            #     torch.load("red_neuronal_comesolo.pth")
            # )  # Carga los pesos

            while not self.verificar_fin_juego():
                origen_optimo, destino_optimo = self.obtener_movimiento_optimo(
                    self.tablero
                )

                self.realizar_movimiento((origen_optimo, destino_optimo))
                self.imprimir_tablero()

            print("¡Has ganado!")

            # Pregunta si el usuario quiere jugar otra ronda
            jugar_otra_vez = input("¿Quieres jugar otra ronda? (s/n): ")
            if jugar_otra_vez.lower() != "s":
                break

    def verificar_fin_juego(self):
        fichas_restantes = sum(1 for ficha in self.tablero if ficha != 0)
        return fichas_restantes == 1


if __name__ == "__main__":
    juego = Comesolo()
    juego.entrenar(1000)
    juego.jugar()
