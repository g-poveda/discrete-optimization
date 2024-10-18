use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Knapsack {
    weights: Vec<u32>,  // List of weights
    profits: Vec<u32>,  // List of profits
    capacity: u32,      // Maximum weight capacity of the knapsack
}

#[pymethods]
impl Knapsack {
    #[new]
    pub fn new(weights: Vec<u32>, profits: Vec<u32>, capacity: u32) -> Self {
        Knapsack { weights, profits, capacity }
    }
    // A method to calculate profit-to-weight ratio for each item
    pub fn ratios(&self) -> Vec<f32> {
        self.profits.iter()
            .zip(&self.weights)
            .map(|(p, w)| *p as f32 / *w as f32)
            .collect()
    }

     // Greedy heuristic to solve the knapsack problem based on profit-to-weight ratio
    pub fn greedy_solution(&self) -> (Vec<u32>, u32, u32) {
        let mut items: Vec<(usize, f32)> = self.ratios()  // (index, ratio)
            .into_iter()
            .enumerate() // Attach original index to each ratio
            .collect();

        // Sort items by their profit-to-weight ratio in descending order
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut total_weight = 0;
        let mut total_profit = 0;
        let mut selected_items = vec![0; self.weights.len()];  // Track selected items as 0 (not taken) or 1 (taken)

        for (i, _) in items {
            let item_weight = self.weights[i];
            let item_profit = self.profits[i];

            if total_weight + item_weight <= self.capacity {
                selected_items[i] = 1;  // Mark item as selected
                total_weight += item_weight;
                total_profit += item_profit;
            }
        }

        (selected_items, total_weight, total_profit)
    }


    // Dynamic programming solution for the 0/1 knapsack problem
    pub fn dp_solution(&self) -> (Vec<u32>, u32, u32) {
        let n = self.weights.len();
        let capacity = self.capacity;

        // Create a DP table with (n + 1) x (capacity + 1)
        let mut dp = vec![vec![0; (capacity + 1) as usize]; (n + 1) as usize];

        // Build the DP table
        for i in 1..=n {
            for w in 0..=capacity {
                if self.weights[i - 1] <= w {
                    // If the weight of the current item is less than or equal to the current weight w
                    dp[i][w as usize] = dp[i - 1][w as usize].max(dp[i - 1][(w - self.weights[i - 1]) as usize] + self.profits[i - 1]);
                } else {
                    // If the weight of the current item is more than w, we can't include it
                    dp[i][w as usize] = dp[i - 1][w as usize];
                }
            }
        }

        // Now dp[n][capacity] contains the maximum profit
        let total_profit = dp[n][capacity as usize];

        // Backtrack to find which items to include in the optimal solution
        let mut total_weight = 0;
        let mut selected_items = vec![0; n];  // Track selected items as 0 (not taken) or 1 (taken)

        let mut w = capacity;
        for i in (1..=n).rev() {
            if dp[i][w as usize] != dp[i - 1][w as usize] {
                // This item was included
                selected_items[i - 1] = 1;  // Mark item as selected
                total_weight += self.weights[i - 1];
                w -= self.weights[i - 1];   // Reduce the remaining weight
            }
        }

        (selected_items, total_weight, total_profit)
    }
}

// /// A Python module implemented in Rust.
// #[pymodule]
// fn do_algorithms(py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<Knapsack>()?;
//     Ok(())
// }
