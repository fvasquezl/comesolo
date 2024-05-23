import tensorflow as tf
import numpy as np
import random


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
                print(f"\033[33m{self.tablero[i]}\033[0m   ", end="")  # Amarillo
            else:
                print(f"{self.tablero[i]}   ", end="")
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

    def realizar_movimiento(self, movimiento=(0, 0)):
        origen, destino = movimiento

        print(origen, destino)

        if not self.movimiento_valido(origen, destino):
            print("movimiento invalido")
            return
        self.tablero[destino - 1] = self.tablero[origen - 1]
        self.tablero[origen - 1] = 0
        self.tablero[((origen + destino) // 2) - 1] = 0

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

    def juega_movimiento_aleatorio(self):
        """Juega un movimiento aleatorio válido."""
        movimientos = self.movimientos_posibles()
        if movimientos:
            movimiento = random.choice(movimientos)
            self.realizar_movimiento(movimiento)


# Define la función de recompensa
def recompensa(tablero, fichas_IA):
    """Devuelve la recompensa para el agente."""
    if tablero.count(0) == 0:  # Si el agente ganó
        return 1
    elif tablero.count(0) == 1:  # Si el agente perdió
        return -1
    else:
        return 0


# Crea la red neuronal

modelo = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(15,)),
        tf.keras.layers.Dense(15, activation="softmax"),
    ]
)

# Compila el modelo
modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Función para entrenar el modelo
def entrenar(modelo, num_iteraciones):

    for i in range(num_iteraciones):
        juego = Comesolo()
        estado = juego.estado_a_vector()

        # Movimiento inicial del jugador (simulando al jugador)
        primer_movimiento = random.randint(1, 15)
        juego.primer_movimiento(primer_movimiento)

        origen = random.randint(1, 15)
        destino = random.randint(1, 15)
        juego.realizar_movimiento((origen, destino))
        estado = juego.estado_a_vector()
        fichas_IA = juego.tablero.count(0)

        while True:
            # Obtiene las probabilidades de las acciones del modelo
            probabilidades = modelo(np.expand_dims(estado, axis=0))

            # Selecciona una acción (con la mayor probabilidad)
            accion = np.argmax(probabilidades)

            # Encuentra el movimiento válido
            movimientos = juego.movimientos_posibles()
            if movimientos:
                for movimiento in movimientos:
                    if movimiento[0] == accion + 1:
                        juego.realizar_movimiento(movimiento)
                        break

            # Calcula la recompensa
            recompensa_actual = recompensa(juego.tablero, fichas_IA)

            # Actualiza el modelo (proceso de aprendizaje)
            # Convierte la acción en un vector one-hot
            target = np.zeros(15)
            target[accion] = 1
            modelo.fit(
                np.expand_dims(estado, axis=0),
                np.expand_dims(target, axis=0),
                epochs=1,
                verbose=0,
            )

            # Si la partida termina, sale del bucle
            if recompensa_actual != 0:
                break

            # Actualiza el estado del juego
            estado = juego.estado_a_vector()
            fichas_IA = juego.tablero.count(0)

        # Imprime el progreso del entrenamiento (opcional)
        print(f"Iteración {i+1}")


# Función para tomar una decisión (usando el modelo entrenado)
def tomar_decision(modelo, estado):
    """Utiliza el modelo para predecir el mejor movimiento."""
    probabilidades = modelo(np.expand_dims(estado, axis=0))
    mejor_movimiento = (
        np.argmax(probabilidades) + 1
    )  # +1 para convertir el índice a la posición del tablero
    return mejor_movimiento


if __name__ == "__main__":
    juego = Comesolo()

    # Entrena el modelo
    entrenar(modelo, 1000)  # Entrena durante 1000 iteraciones

    # Simula el primer movimiento del jugador
    primer_movimiento = int(
        input("Ingresa el número de la posición del primer movimiento: ")
    )
    juego.primer_movimiento(primer_movimiento)

    while True:
        juego.imprimir_tablero()

        if juego.movimientos_posibles():
            # Movimiento de la IA
            origen = 11  # Puedes cambiarlo por una variable
            destino = tomar_decision(modelo, juego.estado_a_vector(), origen)
            if destino:
                juego.realizar_movimiento((origen, destino))
        else:
            print("La IA no puede hacer un movimiento.")
            break

        if juego.tablero.count(0) == 0 or juego.tablero.count(0) == 1:
            break

    juego.imprimir_tablero()
