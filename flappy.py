import sys
import random
import pygame
import neat

pygame.init()
pygame.display.set_caption("FLAPPY BIRD")
WINDOW_SIZE = 500, 750
CLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode(WINDOW_SIZE)
BACKGROUND = pygame.image.load("background.jpg").convert_alpha()
BIRD_SIZE = 45, 32
BIRD_CENTER = WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2
BIRD_IMG = pygame.image.load("bird.png").convert_alpha()
BIRD_IMG = pygame.transform.scale(BIRD_IMG, BIRD_SIZE)
GRAVITY = 5
JUMP = 60
PIPE_INITIAL_X = 700
PIPE_IMG = pygame.image.load("pipe.png").convert_alpha()
PIPE_IMG = pygame.transform.scale2x(PIPE_IMG)
PIPE_FLIP_IMG = pygame.transform.flip(PIPE_IMG, False, True)
PIPE_HEIGHTS = [150, 200, 250, 300, 350]
GAP_PIPE = 200
PIPE_EVENT = pygame.USEREVENT
pygame.time.set_timer(PIPE_EVENT, 800)
FLOOR_Y = 640
FONT = pygame.font.SysFont("bahnschrift", 15)
DELTA_SCORE = 0.1
GENERATION = 0
HIGH_SCORE = 0

class Pipe:
    def __init__(self, height, bottom=True):
        self.bottom = bottom
        if bottom:
            pipe_midtop = PIPE_INITIAL_X, WINDOW_SIZE[1]-height
            self.pipe = PIPE_IMG.get_rect(midtop=pipe_midtop)
        else:
            pipe_midbottom = PIPE_INITIAL_X, height
            self.pipe = PIPE_IMG.get_rect(midbottom=pipe_midbottom)
    
    def display_pipe(self):
        if self.bottom:
            SCREEN.blit(PIPE_IMG, self.pipe)
        else:
            SCREEN.blit(PIPE_FLIP_IMG, self.pipe)

class Bird:
    def __init__(self):
        self.bird_rect = BIRD_IMG.get_rect(center=BIRD_CENTER)
        self.dead = False
        self.score = 0
    
    def collision(self, pipes):
        for curr_pipe in pipes:
            if self.bird_rect.colliderect(curr_pipe.pipe):
                return True
        if self.bird_rect.midbottom[1] >= FLOOR_Y or self.bird_rect.midtop[1] < 0:
            return True
        return False
    
    def get_nearest_pipes(self, pipes):
        nearest_pipe_top = None
        nearest_pipe_bottom = None
        min_distance = 1000
        for pipe in pipes:
            curr_distance = pipe.pipe.topright[0] - self.bird_rect.topleft[0]
            if curr_distance < 0:
                continue
            if curr_distance < min_distance:
                min_distance = curr_distance
                nearest_pipe_bottom = pipe
            elif curr_distance == min_distance:
                nearest_pipe_top = pipe
        return nearest_pipe_top, nearest_pipe_bottom

    def get_distances(self, top_pipe, bottom_pipe, draw_lines=False):
        distances = []
        distances.append(top_pipe.pipe.bottomright[0] - self.bird_rect.topleft[0])
        distances.append(self.bird_rect.topleft[1] - top_pipe.pipe.bottomright[1])
        distances.append(bottom_pipe.pipe.topright[1] - self.bird_rect.bottomright[1])
        if draw_lines:
            pygame.draw.line(SCREEN, (255, 0, 0), self.bird_rect.midright, top_pipe.pipe.bottomright, 5)
            pygame.draw.line(SCREEN, (255, 0, 0), self.bird_rect.midright, bottom_pipe.pipe.topright, 5)
        return distances



def run(genomes, config):
    global GENERATION, HIGH_SCORE
    GENERATION += 1
    models = []
    birds = []
    pipe_list = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        models.append(net)
        genome.fitness = 0
        birds.append(Bird())

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == PIPE_EVENT:
                bottom_height = random.choice(PIPE_HEIGHTS)
                pipe_list.append(Pipe(bottom_height))
                top_height = WINDOW_SIZE[1] - bottom_height - GAP_PIPE
                pipe_list.append(Pipe(top_height, False))
            
        SCREEN.blit(BACKGROUND, (0, 0))
        to_be_removed_pipes = []
        for pipe in pipe_list:
            pipe.pipe.centerx -=5
            pipe.display_pipe()
            if pipe.pipe.topright[0] < 0:
                to_be_removed_pipes.append(pipe)
        for rem in to_be_removed_pipes:
            pipe_list.remove(rem)

        alive_birds = 0
        max_score = 0

        for i, bird in enumerate(birds):
            if not bird.dead:
                alive_birds += 1
                bird.bird_rect.centery += GRAVITY
                bird.score += DELTA_SCORE
                max_score = max(max_score, bird.score)
                genomes[i][1].fitness += bird.score
                SCREEN.blit(BIRD_IMG, bird.bird_rect)
                bird.dead = bird.collision(pipe_list)
                nearest_pipes = bird.get_nearest_pipes(pipe_list)
                if nearest_pipes[0]:
                    distances = bird.get_distances(*nearest_pipes, True)
                else:
                    distances = [1000, 1000, 1000]
                output = models[i].activate(distances)
                choice = output.index(max(output))
                if choice == 0:
                    bird.bird_rect.centery -= JUMP
        
        if alive_birds == 0:
            return

        HIGH_SCORE = max(HIGH_SCORE, max_score)
        msg = "Generation: {}, Birds Alive: {}, Current Gen Score: {}, High Score: {}".format(GENERATION, alive_birds, int(max_score), int(HIGH_SCORE))
        text = FONT.render(msg, True, (255, 255, 255))
        SCREEN.blit(text, (0, 20))
        pygame.display.update()
        CLOCK.tick(60)

neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "config.txt")
population = neat.Population(neat_config)
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.run(run, 500)