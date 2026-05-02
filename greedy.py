import random
import copy


class traffic:

    # Initialize available directions in the intersection
    def __init__(self):
        self.directions = ["north", "south", "east", "west"]

    # Generate random traffic data for each direction
    def generate_traffic(self):
        traffic_data = {}

        for d in self.directions:
            traffic_data[d] = {
                "cars": random.randint(0, 40),
                "emergency": random.choice([False, False, False, True])
            }

        return traffic_data

    # Apply greedy decision to choose best direction
    def greedy_decision(self, traffic_data):

        # Check if any emergency vehicle exists
        for direction in traffic_data:
            if traffic_data[direction]["emergency"]:
                return direction, "Emergency vehicle detected"

        # Check if all directions have no traffic
        if all(traffic_data[d]["cars"] == 0 for d in traffic_data):
            return None, "No traffic detected"

        # Select direction with highest number of cars
        best_direction = max(
            traffic_data,
            key=lambda d: traffic_data[d]["cars"]
        )

        # Handle tie cases by selecting first match
        max_value = traffic_data[best_direction]["cars"]

        candidates = [
            d for d in traffic_data
            if traffic_data[d]["cars"] == max_value
        ]

        return candidates[0], "Highest traffic selected"

    # Calculate total waiting time in the intersection
    def calculate_wait_time(self, traffic_data):
        total = 0

        for d in traffic_data:
            total = total + traffic_data[d]["cars"]

        return total

    # Simulate traffic improvement after signal decision
    def apply_optimization(self, traffic_data, selected_direction):

        new_data = copy.deepcopy(traffic_data)

        # If no direction selected, return unchanged data
        if selected_direction is None:
            return new_data

        # Reduce traffic in selected direction
        if selected_direction in new_data:
            reduction = random.randint(10, 25)
            new_data[selected_direction]["cars"] = new_data[selected_direction]["cars"] - reduction

            # Ensure no negative values
            if new_data[selected_direction]["cars"] < 0:
                new_data[selected_direction]["cars"] = 0

        return new_data

    # Run full simulation for multiple steps
    def run_simulation(self, steps):

        i = 0

        while i < steps:

            # Generate traffic for current step
            traffic_data = self.generate_traffic()

            # Calculate congestion before optimization
            wait_before = self.calculate_wait_time(traffic_data)

            # Apply greedy algorithm decision
            selected_direction, reason = self.greedy_decision(traffic_data)

            # Apply traffic optimization effect
            optimized = self.apply_optimization(traffic_data, selected_direction)

            # Calculate congestion after optimization
            wait_after = self.calculate_wait_time(optimized)

            # Print simulation results
            print("Step", i + 1)

            print("Traffic before optimization")
            for d in traffic_data:
                print(d, traffic_data[d])

            print("Selected direction", selected_direction)
            print("Reason", reason)

            print("Traffic after optimization")
            for d in optimized:
                print(d, optimized[d])

            print("Wait time before", wait_before)
            print("Wait time after", wait_after)
            print("Improvement", wait_before - wait_after)

            i = i + 1


if __name__ == "__main__":
    system = traffic()
    system.run_simulation(5)