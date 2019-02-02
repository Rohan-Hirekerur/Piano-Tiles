import pygame
import pyglet
from collections import deque
import random
import math
import numpy as np
import cnn
import tensorflow as tf


state_size = [84, 84, 4]
action_size = 5
learning_rate = 0.0002      # Alpha (aka learning rate)

# TRAINING HYPER PARAMETERS
total_episodes = 50000        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 5

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyper parameters
gamma = 0.95               # Discounting rate

# MEMORY HYPER PARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False
testing = True


possible_actions = [0, 1, 2, 3, 4]


white = (255, 255, 255)
blue = (0, 0, 255)
black = (0, 0, 0)
red = (255, 0, 0)

pygame.init()
screen_width = 400
screen_height = 800
size = screen_width, screen_height
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Piano tiles")
pygame.display.update()
clock = pyglet.clock.Clock()
clock.set_fps_limit(60)
font = pygame.font.Font(None, 50)

_songs = ['./mp3/a1.mp3', './mp3/a1s.mp3', './mp3/b1.mp3', './mp3/c1.mp3', './mp3/c1s.mp3', './mp3/c2.mp3', './mp3/d1.mp3',
          './mp3/d1s.mp3', './mp3/e1.mp3', './mp3/f1.mp3', './mp3/f1s.mp3', './mp3/g1.mp3', './mp3/g1s.mp3']


class Tile:
    def __init__(self, x, y):
        self.is_note = False
        self.width = screen_width/4
        self.height = screen_height/4
        self.is_clicked = False
        self.x = x
        self.y = y

    def click(self):
        self.is_clicked = True
        # note = random.choice(_songs)
        # pygame.mixer.music.stop()
        # pygame.mixer.music.load(note)
        # pygame.mixer.music.play()

    def display(self):
        if not self.is_note:
            if not self.is_clicked:
                pygame.draw.rect(screen, white, [self.x, self.y, self.width, self.height])
            else:
                pygame.draw.rect(screen, red, [self.x, self.y, self.width, self.height])
        else:
            if not self.is_clicked:
                pygame.draw.rect(screen, black, [self.x, self.y, self.width, self.height])
            else:
                pygame.draw.rect(screen, white, [self.x, self.y, self.width, self.height])


class Row:
    def __init__(self, y):
        self.pos = random.randint(0, 3)
        self.tiles = []
        self.y = y
        for i in range(0, 4):
            tile = Tile(i*screen_width/4, self.y)
            self.tiles.append(tile)
            if i == self.pos:
                self.tiles[i].is_note = True

    def move(self, speed):
        self.y += speed
        for tile in self.tiles:
            tile.y = self.y

    def display(self):
        for tile in self.tiles:
            tile.display()

    def click(self, x):
        x = math.floor(4*x/screen_width)
        self.tiles[x].click()

    def is_complete(self):
        for i in range(0, 4):
            if self.tiles[i].is_note and not self.tiles[i].is_clicked:
                return False
        return True

    def false_clicked(self):
        for i in range(0, 4):
            if not self.tiles[i].is_note and self.tiles[i].is_clicked:
                return True
        return False


class Grid:
    def __init__(self):
        self.rows = deque()
        self.speed = 5
        for i in range(0, 5):
            y = (4-i) * screen_height / 4 - screen_height / 4
            row = Row(y)
            self.rows.append(row)

    def display(self, score):
        for row in self.rows:
            row.display()
        pygame.draw.line(screen, blue, (0, 7*screen_height/8), (screen_width, 7*screen_height/8), 2)
        text = font.render(str(score), 1, red)
        text_pos = (screen_width/2 - 10, 50)
        screen.blit(text, text_pos)

    def move_rows(self, inc):
        for row in self.rows:
            if row.false_clicked():
                return False
        if self.rows[0].y >= screen_height:
            complete = self.rows[0].is_complete()
            if complete:
                y = self.rows[4].y - screen_height / 4
                row = Row(y)
                self.rows.__delitem__(0)
                self.rows.append(row)
            return complete

        if inc and self.speed < 15:
            self.speed *= 1.05
        for row in self.rows:
                row.speed = self.speed
        for row in self.rows:
            row.move(self.speed)
        return True

    def click(self, pos):
        x = pos[0]
        y = pos[1]
        for row in self.rows:
            if y >= row.y:
                row.click(x)
                break


def preprocess_frame(frame):
    # Greyscale frame
    frame = np.mean(frame, -1)

    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = np.resize(normalized_frame, [84, 84])

    return preprocessed_frame


stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = cnn.Cnn(state_size, action_size, learning_rate)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


memory = Memory(max_size=memory_size)
if training:
    for i in range(pretrain_length):
        # If it's the first step
        # First we need a state
        grid = Grid()
        start = False
        inc = False
        complete = True
        score = 0
        reward = 0
        grid.display(score)
        pygame.display.update()
        state = pygame.surfarray.array3d(screen)
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        # Random action
        action = random.choice(possible_actions)


        while complete:
            clock.tick()
            screen.fill(blue)
            events = pygame.event.get()
            reward += 1

            if action != 0:
                if action == 1:
                    pos = (screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1
                if action == 2:
                    pos = (3 * screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1
                if action == 3:
                    pos = (5 * screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1
                if action == 4:
                    pos = (7 * screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1

                if score % 5 == 0:
                    inc = True

            complete = grid.move_rows(inc)
            inc = False
            grid.display(score)

            # If we're dead
            if not complete:
                # We finished the episode
                reward -= 1000
                next_state = np.zeros(state.shape)

                # Add experience to memory
                memory.add((state, action, reward, next_state, not complete))
            else:
                # Get the next state
                pygame.display.update()
                next_state = pygame.surfarray.array3d(screen)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Add experience to memory
                memory.add((state, action, reward, next_state, not complete))

                # Our state is now the next_state
                state = next_state


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


# Saver will help us to save our model
saver = tf.train.Saver()

if training:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state

            grid = Grid()
            start = False
            inc = False
            complete = True
            score = 0
            reward = 0

            grid.display(score)
            pygame.display.update()
            state = pygame.surfarray.array3d(screen)

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while complete:
                clock.tick()
                step += 1

                # Increase decay_step
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step,
                                                             state,
                                                             possible_actions)

                screen.fill(blue)
                events = pygame.event.get()
                reward += 1

                if action != 0:
                    if action == 1:
                        pos = (screen_width / 8, 7 * screen_height / 8)
                        grid.click(pos)
                        score += 1
                        reward += 10
                    if action == 2:
                        pos = (3 * screen_width / 8, 7 * screen_height / 8)
                        grid.click(pos)
                        score += 1
                        reward += 10
                    if action == 3:
                        pos = (5 * screen_width / 8, 7 * screen_height / 8)
                        grid.click(pos)
                        score += 1
                        reward += 10
                    if action == 4:
                        pos = (7 * screen_width / 8, 7 * screen_height / 8)
                        grid.click(pos)
                        score += 1
                        reward += 10

                    if score % 5 == 0:
                        inc = True

                complete = grid.move_rows(inc)
                inc = False
                grid.display(score)


                # Look if the episode is finished
                done = not complete

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    reward -= 1000
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                           #'Training loss: {}'.format(sess.run(DQNetwork.loss)),
                          'Explore P: {:.4f}'.format(explore_probability))

                    memory.add((state, action, reward, next_state, done))

                else:
                    # Get the next state
                    pygame.display.update()
                    next_state = pygame.surfarray.array3d(screen)

                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs: states_mb,
                                              DQNetwork.sample_op: targets_mb,
                                              DQNetwork.actions: actions_mb})
                print(loss)

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")


if testing:
    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, "./models/model.ckpt")
        for i in range(1):

            done = False

            grid = Grid()
            start = False
            inc = False
            complete = True
            score = 0

            grid.display(score)
            pygame.display.update()
            state = pygame.surfarray.array3d(screen)
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while complete:
                clock.tick()
                screen.fill(blue)
                events = pygame.event.get()
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs: state.reshape((1, *state.shape))})

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]

                if action == 1:
                    pos = (screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1
                    print("g")
                if action == 2:
                    pos = (3 * screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1
                if action == 3:
                    pos = (5 * screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1
                if action == 4:
                    pos = (7 * screen_width / 8, 7 * screen_height / 8)
                    grid.click(pos)
                    score += 1

                    if score % 5 == 0:
                        inc = True

                complete = grid.move_rows(inc)
                inc = False
                grid.display(score)
                pygame.display.update()
pygame.time.delay(3000)
pygame.quit()
