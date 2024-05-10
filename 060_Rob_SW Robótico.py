#Salazar Chavez Cristian Uriel
#21310215
import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # Costo desde el nodo inicial hasta este nodo
        self.h = 0  # Heurística: estimación del costo desde este nodo hasta el objetivo
        self.f = 0  # Costo total: g + h
    
    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def a_star(grid, start, goal):
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, start)
    
    while open_set:
        current = heapq.heappop(open_set)
        
        if current.x == goal.x and current.y == goal.y:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]
        
        closed_set.add((current.x, current.y))
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = current.x + dx, current.y + dy
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] != 1 and (new_x, new_y) not in closed_set:
                new_node = Node(new_x, new_y, current)
                new_node.g = current.g + 1
                new_node.h = heuristic(new_node, goal)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(open_set, new_node)
    
    return None

# Ejemplo de mapa en una cuadrícula (0: camino libre, 1: obstáculo)
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

# Punto de inicio y punto objetivo
start = Node(0, 0)
goal = Node(4, 4)

# Buscar el camino más corto utilizando A*
path = a_star(grid, start, goal)

# Visualizar el camino encontrado
if path:
    print("Camino encontrado:", path)
else:
    print("No se encontró un camino posible.")
