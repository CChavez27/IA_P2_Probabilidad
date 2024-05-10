#Salazar Chavez Cristian Uriel
#21310215 

import pygame
import random

# Inicializar pygame
pygame.init()

# Definir colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Definir tamaño de la ventana
WIDTH, HEIGHT = 800, 600

# Clase para el robot
class Robot(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speed = 3

    def update(self):
        # Movimiento aleatorio del robot
        direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        if direction == 'UP':
            self.rect.y -= self.speed
        elif direction == 'DOWN':
            self.rect.y += self.speed
        elif direction == 'LEFT':
            self.rect.x -= self.speed
        elif direction == 'RIGHT':
            self.rect.x += self.speed

        # Verificar límites de la ventana
        self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))
        self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))

# Configurar la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("HW Robótico: Sensores y Actuadores")

clock = pygame.time.Clock()

# Crear el robot
robot = Robot(WIDTH // 2, HEIGHT // 2)

# Crear grupo de sprites y agregar el robot
all_sprites = pygame.sprite.Group()
all_sprites.add(robot)

# Bucle principal
running = True
while running:
    # Manejo de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Actualizar
    all_sprites.update()

    # Renderizar
    screen.fill(WHITE)
    all_sprites.draw(screen)

    # Refrescar la pantalla
    pygame.display.flip()

    # Controlar la velocidad de la actualización
    clock.tick(60)

# Salir del programa
pygame.quit()
