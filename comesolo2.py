class Comesolo:
    def __init__(self):
        self.tablero = [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15]]
        self.estado = 0

    def inc_estado(self):
        self.estado += 1

    def imprimir_tablero(self) -> None:
        """
        Imprime el estado actual del tablero de manera formateada.
        """
        print(f"Estado de tablero: {self.estado}")
        for fila in self.tablero:
            print(" " * (5 - len(fila)), end="")
            for valor in fila:
                if valor == 0:
                    print(f"\033[33m{valor:1d} \033[0m ", end="")  # Amarillo
                else:
                    print(f"{valor:1d} ", end="")
            print()
        self.inc_estado()

    def movimiento_valido(self, origen, destino):
        """
        Verifica si un movimiento es válido según las reglas del juego.

        Args:
            origen (int): Índice de la ficha de origen (empezando desde 1).
            destino (int): Índice de la posición de destino (empezando desde 1).

        Returns:
            bool: True si el movimiento es válido, False en caso contrario.
        """
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

    def obtener_coordenadas(self, indice):
        """
        Obtiene las coordenadas de fila y columna a partir del índice de una ficha.

        Args:
            indice (int): Índice de la ficha (empezando desde 1).

        Returns:
            tuple: Tupla con las coordenadas de fila y columna.
        """
        for fila in range(len(self.tablero)):
            for col in range(len(self.tablero[fila])):
                if self.tablero[fila][col] == indice:
                    return fila, col

    def realizar_movimiento(self, origen, destino) -> None:
        """
        Realiza un movimiento válido en el tablero.

        Args:
            origen (int): Índice de la ficha de origen (empezando desde 1).
            destino (int): Índice de la posición de destino (empezando desde 1).
        """
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

    def jugar(self):
        """
        Inicia el juego y permite al usuario realizar el primer movimiento.
        """
        self.imprimir_tablero()
        primer_movimiento = int(
            input(
                "Ingresa el índice de la ficha a eliminar para iniciar el juego (1-15): "
            )
        )
        origen, destino = self.obtener_coordenadas(primer_movimiento)
        self.tablero[origen][destino] = 0
        self.imprimir_tablero()

        # Aquí debes implementar la lógica de la IA para resolver el juego
        while not self.verificar_fin_juego():
            # Obtener el siguiente movimiento óptimo utilizando tu algoritmo de IA
            origen_optimo, destino_optimo = obtener_movimiento_optimo(self.tablero)
            self.realizar_movimiento(origen_optimo, destino_optimo)
            self.imprimir_tablero()

        print("¡Has ganado!")

    def verificar_fin_juego(self):
        """
        Verifica si el juego ha terminado (solo queda una ficha en el tablero).

        Returns:
            bool: True si el juego ha terminado, False en caso contrario.
        """
        fichas_restantes = sum(fila.count(0) for fila in self.tablero)
        return fichas_restantes == len(self.tablero) * (len(self.tablero) + 1) // 2 - 1


if __name__ == "__main__":
    juego = Comesolo()
    juego.jugar()
