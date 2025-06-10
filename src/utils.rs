
use ndarray::{Array, ArrayView, Axis, Dim, IntoDimension, IxDyn, s};

// Finds the index of the maximum value in a 2D tensor
pub fn argmax(tensor: &ArrayView<f32, IxDyn>) -> (usize, usize) {
    let mut max_index = (0, 0);
    let mut max_value = tensor
        .get(IxDyn::zeros(2))
        .copied()
        .unwrap_or(f32::NEG_INFINITY);

    for i in 0..tensor.shape()[0] {
        for j in 0..tensor.shape()[1] {
            if let Some(value) = tensor.get((i, j).into_dimension()) {
                if *value > max_value {
                    max_value = *value;
                    max_index = (i, j);
                }
            }
        }
    }
    max_index
}


// pub fn argmax(tensor: &ArrayView<f32, IxDyn>) -> (usize, usize) {
//     let mut max_index = (0, 0);
//     let mut max_value = tensor
//         .get(IxDyn::zeros(2))
//         .copied()
//         .unwrap_or(f32::NEG_INFINITY);

//     for i in 0..tensor.shape()[0] {
//         for j in 0..tensor.shape()[1] {
//             if let Some(value) = tensor.get((i, j).into_dimension()) {
//                 if *value > max_value {
//                     max_value = *value;
//                     max_index = (i, j);
//                 }
//             }
//         }
//     }
//     max_index
// }
