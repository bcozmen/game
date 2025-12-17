import math
import pygame
HEX_SIZE = 30
WIDTH, HEIGHT = 800, 600


class VecMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = self.create_vec_map()

    def create_vec_map(self):
        self.map = torch.zeros((self.height, self.width, 2))

class HexTile:
    def __init__(self, x, y, terrain):
        self.x = x
        self.y = y
        self.terrain = terrain


class HexMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = self.create_hex_map()

    def create_hex_map(self):
        hex_map = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                row.append(self.create_hex_tile(x, y))
            hex_map.append(row)
        return hex_map

    def create_hex_tile(self, x, y):
        return HexTile(x, y, terrain='grass')

    def display_map(self):
        for row in self.map:
            print(' '.join(f"({tile.x},{tile.y})" for tile in row))

    def hex_neighbors(self, x, y):
        if y % 2 == 0:  # even row
            directions = [
                (+1, 0), (0, +1), (-1, +1),
                (-1, 0), (-1, -1), (0, -1)
            ]
        else:  # odd row
            directions = [
                (+1, 0), (+1, +1), (0, +1),
                (-1, 0), (0, -1), (+1, -1)
            ]

        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append(self.map[ny][nx])
        return neighbors

    def hex_to_pixel(self, x, y):
        """Odd-row offset, pointy-top"""
        px = HEX_SIZE * math.sqrt(3) * (x + 0.5 * (y % 2))
        py = HEX_SIZE * 1.5 * y
        return int(px + 50), int(py + 50)

    def draw_hex(self, surface, color, center):
        cx, cy = center
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            px = cx + HEX_SIZE * math.cos(angle)
            py = cy + HEX_SIZE * math.sin(angle)
            points.append((px, py))
        pygame.draw.polygon(surface, color, points, 1)

    def run_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()

        running = True
        while running:
            clock.tick(30)
            screen.fill((30, 30, 30))

            for y in range(self.height):
                for x in range(self.width):
                    tile = self.map[y][x]
                    pos = self.hex_to_pixel(tile.x, tile.y)
                    self.draw_hex(screen, (0, 200, 0), pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()

        pygame.quit()



if __name__ == "__main__":
    hex_map = HexMap(50, 50)
    hex_map.display_map()
    hex_map.run_pygame()