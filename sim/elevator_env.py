import random

class ElevatorEnv:
    def __init__(self, num_floors=5):
        self.num_floors = num_floors
        self.reset()

    def reset(self):
        self.current_floor = 0
        self.direction = 1  # 1 = up, -1 = down
        self.requests = self.generate_requests()
        self.total_wait_time = 0
        self.steps = 0

        return self.get_state()

    def generate_requests(self):
        requests = []

        for _ in range(random.randint(2, 5)):
            source = random.randint(0, self.num_floors - 1)
            dest = random.randint(0, self.num_floors - 1)

            while dest == source:
                dest = random.randint(0, self.num_floors - 1)

            requests.append((source, dest))

        return requests

    def get_state(self):

        request_floors = tuple(
            sorted([req[0] for req in self.requests])
        )

        return (
            self.current_floor,
            request_floors,
            self.direction
        )

    def step(self, action):

        reward = -2

        # ACTIONS
        # 0 = UP
        # 1 = DOWN
        # 2 = OPEN DOOR
        # 3 = IDLE

        moved = False

        if action == 0:
            if self.current_floor < self.num_floors - 1:
                self.current_floor += 1
                self.direction = 1
                moved = True
            else:
                reward -= 5

        elif action == 1:
            if self.current_floor > 0:
                self.current_floor -= 1
                self.direction = -1
                moved = True
            else:
                reward -= 5

        elif action == 2:

            served_requests = []

            for req in self.requests:

                source, dest = req

                if source == self.current_floor:
                    served_requests.append(req)

            if served_requests:

                reward += 20

                for req in served_requests:
                    self.requests.remove(req)

            else:
                reward -= 10

        elif action == 3:
            reward -= 8

        # Small movement reward
        if moved:
            reward += 1

        self.steps += 1

        # waiting penalty
        reward -= len(self.requests)

        self.total_wait_time += len(self.requests)

        done = len(self.requests) == 0 or self.steps >= 50

        return self.get_state(), reward, done

    def render(self):
        print(f"\nElevator Floor: {self.current_floor}")
        print(f"Direction: {'UP' if self.direction == 1 else 'DOWN'}")
        print(f"Pending Requests: {self.requests}")

if __name__ == "__main__":
    env = ElevatorEnv()

    state = env.reset()

    done = False

    while not done:
        env.render()

        action = random.randint(0, 3)

        next_state, reward, done = env.step(action)

        print("Action:", action)
        print("Reward:", reward)