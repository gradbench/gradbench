use std::ops::{Index, IndexMut};

use serde::{de::Error as _, Deserialize, Deserializer};

#[derive(Debug)]
pub struct Matrix {
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.; rows * cols];
        Self { cols, data }
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, row: usize) -> &[f64] {
        let i = row * self.cols;
        &self.data[i..i + self.cols]
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        debug_assert!(col < self.cols);
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        debug_assert!(col < self.cols);
        &mut self.data[row * self.cols + col]
    }
}

impl<'de> Deserialize<'de> for Matrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let rows_vec = Vec::<Vec<f64>>::deserialize(deserializer)?;
        if rows_vec.is_empty() {
            return Ok(Matrix {
                cols: 0,
                data: Vec::new(),
            });
        }
        let cols = rows_vec[0].len();
        if cols == 0 {
            return Err(D::Error::custom("matrix has zero columns"));
        }
        if rows_vec.iter().any(|r| r.len() != cols) {
            return Err(D::Error::custom(
                "all rows must have the same number of columns",
            ));
        }
        let data: Vec<f64> = rows_vec.into_iter().flatten().collect();
        Ok(Matrix { cols, data })
    }
}
