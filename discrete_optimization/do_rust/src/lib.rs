use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
pub mod algos; // This line imports the entire `algorithms` module

pub use algos::knapsack::*;// This line pulls the `Knapsack` struct into the root module
pub use algos::coloring::*;

#[pymodule]
fn do_rust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Knapsack>()?;
    m.add_class::<PyGraph>()?;
    Ok(())
}
