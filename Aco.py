import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Ant Colony Optimization for JSSP
def initialize_pheromone(num_operations):
    return np.ones((num_operations, num_operations))

def heuristic_matrix(processing_times):
    return 1 / (processing_times + 1e-6)  # Avoid division by zero

def construct_solution(pheromones, heuristic, num_operations, alpha, beta):
    solution = []
    visited = set()

    for _ in range(num_operations):
        probabilities = []
        for i in range(num_operations):
            if i not in visited:
                prob = (pheromones[i].sum() ** alpha) * (heuristic[i].sum() ** beta)
                probabilities.append(prob)
            else:
                probabilities.append(0)
        
        probabilities = np.array(probabilities)
        if probabilities.sum() > 0:
            probabilities /= probabilities.sum()
        else:
            probabilities = np.ones(num_operations) / num_operations

        next_operation = np.random.choice(range(num_operations), p=probabilities)
        solution.append(next_operation)
        visited.add(next_operation)

    return solution

def evaluate_solution(solution, job_data):
    machine_end_times = {}
    job_end_times = {}

    for operation in solution:
        job, machine, time = job_data[operation]

        start_time = max(job_end_times.get(job, 0), machine_end_times.get(machine, 0))
        end_time = start_time + time

        job_end_times[job] = end_time
        machine_end_times[machine] = end_time

    makespan = max(job_end_times.values())
    return makespan

def update_pheromones(pheromones, solutions, makespans, evaporation_rate):
    pheromones *= (1 - evaporation_rate)

    for solution, makespan in zip(solutions, makespans):
        for i in range(len(solution) - 1):
            pheromones[solution[i], solution[i+1]] += 1 / makespan

    return pheromones

def plot_makespan_evolution(iteration_makespans):
    # Create the line plot for makespan evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(iteration_makespans) + 1), iteration_makespans, marker='o', color='blue')
    ax.set_title("Makespan Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Makespan")
    ax.grid(True)
    plt.tight_layout()

    return fig

# Streamlit App
def main():
    st.title("Ant Colony Optimization for Job Shop Scheduling Problem")

    # Upload JSSP Dataset
    uploaded_file = st.file_uploader("Upload a JSSP dataset (CSV format)", type=["csv"])

    if uploaded_file:
        job_data_df = pd.read_csv(uploaded_file)
        # Extract necessary data: Job ID, Machine ID, Processing Time
        job_data = job_data_df[["Job ID", "Machine ID", "Processing Time"]].values
        num_operations = len(job_data)

        st.write("Uploaded Dataset:")
        st.write(job_data_df)

        # ACO Parameters
        alpha = st.slider("Pheromone Importance (Alpha):", 0.1, 5.0, 1.0)
        beta = st.slider("Heuristic Importance (Beta):", 0.1, 5.0, 2.0)
        evaporation_rate = st.slider("Pheromone Evaporation Rate:", 0.01, 1.0, 0.1)
        num_ants = st.slider("Number of Ants:", 1, 100, 10)
        num_iterations = st.slider("Number of Iterations:", 1, 500, 100)

        # Run ACO
        if st.button("Run ACO"):
            pheromones = initialize_pheromone(num_operations)
            heuristic = heuristic_matrix(job_data[:, 2].astype(float))

            best_solution = None
            best_makespan = float('inf')
            iteration_makespans = []

            for iteration in range(num_iterations):
                solutions = []
                makespans = []

                for _ in range(num_ants):
                    solution = construct_solution(pheromones, heuristic, num_operations, alpha, beta)
                    makespan = evaluate_solution(solution, job_data)

                    solutions.append(solution)
                    makespans.append(makespan)

                    if makespan < best_makespan:
                        best_solution = solution
                        best_makespan = makespan

                pheromones = update_pheromones(pheromones, solutions, makespans, evaporation_rate)
                iteration_makespans.append(best_makespan)

                st.write(f"Iteration {iteration + 1}: Best Makespan = {best_makespan}")

            # Display Results
            st.subheader("Best Solution")
            st.write("Operation Sequence:", best_solution)
            st.write("Makespan:", best_makespan)

            # Plot Makespan Evolution
            st.subheader("Makespan Evolution")
            makespan_fig = plot_makespan_evolution(iteration_makespans)
            st.pyplot(makespan_fig)

if __name__ == "__main__":
    main()
