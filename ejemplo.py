class Operacion:
    def __init__(self, a, b):
        self.parametro_a = a
        self.parametro_b = b
        self.parametro_c = 0

    def suma(self):
        self.parametro_c = self.parametro_a + self.parametro_b
        return self.parametro_c

    def mult(self):
        self.parametro_c = self.parametro_a * self.parametro_b
        return self.parametro_c


if __name__ == "__main__":

    # Jugar el juego
    op = Operacion(5, 8)
    x = op.suma()
    y = op.mult()

    print(x, y)
