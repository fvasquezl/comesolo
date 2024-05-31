import os
from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time


class Comesolo:
    def __init__(self):
        self.tablero = self.ini_tablero()
        self.estado = 0
        self.movimiento_inicial = 0
        self.movimientos = []

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

    def imprimir_solucion(self):
        juego.ini_tablero()
        juego.primer_movimiento(self.movimiento_inicial)
        for movimiento in self.movimientos:
            juego.imprimir_tablero()
            time.sleep(1)
            origen, destino = movimiento
            juego.realizar_movimiento(origen, destino)
            juego.imprimir_tablero()

    def jugar(self):
        juego.ini_tablero()
        self.movimiento_inicial = int(input("Ingrese su movimiento inicial (1-15): "))
        juego.primer_movimiento(self.movimiento_inicial)
        print("Pensando ... :)")
        while True:
            # juego.imprimir_tablero()
            bytes_aleatorios = os.urandom(8)
            # Convertir los bytes aleatorios a un número entero y utilizarlo como semilla
            semilla = int.from_bytes(bytes_aleatorios, byteorder="big")
            random.seed(semilla)

            # juego.imprimir_tablero()
            movimientos_validos = juego.obtener_movimientos_validos()
            if not movimientos_validos:
                if sum(juego.tablero) == 1:
                    break
                else:
                    juego.movimientos = []
                    juego.ini_tablero()
                    juego.primer_movimiento(self.movimiento_inicial)
            else:
                origen, destino = random.choice(movimientos_validos)
                juego.realizar_movimiento(origen, destino)
                self.movimientos.append((origen, destino))

        juego.imprimir_solucion()


if __name__ == "__main__":

    # Jugar el juego
    juego = Comesolo()
    juego.jugar()
