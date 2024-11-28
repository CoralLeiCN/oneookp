use ndarray::Array;
use ndarray::Dimension;
// use ndarray::Shape;
use ndarray::ShapeBuilder;
use num_traits::Zero;
use std::collections::HashMap;
// trait Real {}

// impl Real for f32 {}
// impl Real for f64 {}

//the zero with dim is generic over some type Sh
fn zeros_with_dim<T, Sh>(dim: Sh) -> Array<T, <Sh as ShapeBuilder>::Dim>
where
    T: Zero + Clone,
    Sh: ShapeBuilder<Dim: Dimension>,
{
    // # original codes examples
    // let a = Array::<f64, _>::zeros((1, 3, 2).f());

    // # dimensionality explicitly Array::<f64, Ix3>::zeros(...), withIx3
    // let a = Array::<f64, Ix3>::zeros((1, 3, 2).f());

    // # f() makes a column major, so without f() it will be row major
    // let a = Array::<f64, Ix3>::zeros((1, 3, 2));

    Array::<T, _>::zeros(dim)
}

fn array_value_counts(
    arr: &ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<ndarray::IxDynImpl>>,
) -> HashMap<&i32, i32> {
    let mut map = HashMap::new();
    let flat_view: ndarray::iter::Iter<'_, i32, ndarray::Dim<ndarray::IxDynImpl>> =
        arr.view().into_dyn().into_iter();

    for item in flat_view {
        *map.entry(item).or_insert(0) += 1;
    }
    map
}

fn print_type<T>(_: &T) {
    println!("{:?}", std::any::type_name::<T>());
}

use ndarray::{Array1, Axis};

fn argsort<T: PartialOrd>(arr: &Array1<T>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap());
    indices
}

fn main() {
    // The Array<A, D> is parameterized by A for the element type and D for the dimensionality.
    // through turbofish syntax, and let it infer the dimensionality type
    let a: Array<f32, _> = zeros_with_dim((1, 3, 2));
    println!("{:?}", a);

    let a: Array<f64, _> = zeros_with_dim((1, 3, 2));
    println!("{:?}", a);

    let arr1: ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<ndarray::IxDynImpl>> =
        Array::from_vec(vec![0, 1, 2, 2, 3, 1])
            .into_shape_with_order(ndarray::IxDyn(&[6]))
            .unwrap();
    let arr2: ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<ndarray::IxDynImpl>> =
        Array::from_vec(vec![0, 2, 1, 2, 3, 4])
            .into_shape_with_order(ndarray::IxDyn(&[6]))
            .unwrap();
    let map: HashMap<&i32, i32> = array_value_counts(&arr1);

    println!("{:?}", map);

    let map_unique_key: std::collections::hash_map::Keys<'_, &i32, i32> = map.keys();
    println!("{:?}", map_unique_key);

    let len_map: usize = map.len();
    println!("unique value of dict {:?}", len_map);

    //init empty crosstab
    let mut zero_crosstab: ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<[usize; 2]>> =
        Array::<i32, _>::zeros((len_map + 2, len_map + 2));
    println!("{:?}", zero_crosstab);

    for (i, a) in arr1.iter().enumerate() {
        for (j, b) in arr2.iter().enumerate() {
            zero_crosstab[[i, j]] += 1;
            // println!(
            //     "Index i : {}, Index j : {}, Value: {:?}",
            //     i,
            //     j,
            //     zero_crosstab[[i, j]]
            // );
        }
    }
    println!("{:?}", zero_crosstab);
    let b: (i32, i32, i32) = (1, 3, 2);
    print_type(&b);

    // Basic example
    let arr: ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<[usize; 1]>> =
        Array1::from(vec![3, 1, 4, 2, 3]);
    let perm: Vec<usize> = argsort(&arr);

    println!("Array: {:?}", arr);
    println!("Sorted indices: {:?}", perm);
    // Descending order
    let desc_indices: Vec<_> = perm.into_iter().rev().collect();
    println!("\nDescending indices: {:?}", desc_indices);
    // Print sorted array using indices
    let aux: Vec<_> = desc_indices.iter().map(|&i| arr[i]).collect();
    println!("Sorted array: {:?}", aux);

    // masked the sorted array, if different from previous value
    let masked: Vec<_> = aux
        .iter()
        .scan(None, |prev, &x| {
            let mask = match *prev {
                Some(prev) if prev == x => false,
                _ => {
                    *prev = Some(x);
                    true
                }
            };
            Some(mask)
        })
        .collect();
    println!("Masked array: {:?}", masked);

    // calculate cumulative sum for array masked
    let cumsum: Vec<_> = masked
        .iter()
        .scan(0, |state, &x| {
            *state += x as i32;
            Some(*state)
        })
        .collect();
    println!("Cumulative sum: {:?}", cumsum);

    // -1 for cumsum
    let cumsum_index: Vec<_> = cumsum.iter().map(|&x| x - 1).collect();
    println!("Cumulative sum index: {:?}", cumsum_index);
    // init empety ndarray
    let mut inv_idx = Array::<i32, _>::zeros(arr.len());
    println!("Empty array: {:?}", inv_idx);

    // convert descending indices to array
    let desc_indices: Array1<usize> = Array1::from(desc_indices);

    let inv_idx: Vec<_> = desc_indices.iter().map(|&i| cumsum_index[i]).collect();
    println!("Inverse index: {:?}", inv_idx);

    //init empty crosstab
    let mut zero_crosstab: ndarray::ArrayBase<ndarray::OwnedRepr<i32>, ndarray::Dim<[usize; 2]>> =
        Array::<i32, _>::zeros((len_map, len_map));
    for ((i, a), (j, b)) in inv_idx.iter().enumerate().zip(inv_idx.iter().enumerate()) {
        zero_crosstab[(*a as usize, *b as usize)] += 1;
        println!("Index i : {}, Index j : {}", a, b)
    }

    println!("{:?}", zero_crosstab);

    let a = Array::range(0., 10., 1.);

    let mut a = a.mapv(|a: f64| a.powi(3)); // numpy equivlant of `a ** 3`; https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.powi

    println!("{}", a);

    use ndarray::prelude::*;
    println!("{}", a[[2]]);
    println!("{}", a.slice(s![2]));

    let a = array![[3., 7., 3., 4.], [1., 4., 2., 2.], [7., 2., 4., 9.]];

    println!("a = \n{:?}\n", a);

    // use trait FromIterator to flatten a matrix to a vector
    let b = Array::from_iter(a.iter());
    println!("b = \n{:?}\n", b);

    let c = b.into_shape_with_order([6, 2]).unwrap(); // consume b and generate c with new shape
    println!("c = \n{:?}", c);

    let mut a = Array::range(0., 12., 1.)
        .into_shape_with_order([3, 4])
        .unwrap();
    println!("a = \n{}\n", a);

    {
        let (s1, s2) = a.view().split_at(Axis(1), 2);

        // with s as a view sharing the ref of a, we cannot update a here
        // a.slice_mut(s![1, 1]).fill(1234.);

        println!("Split a from Axis(0), at index 1:");
        println!("s1  = \n{}", s1);
        println!("s2  = \n{}\n", s2);
    }
    {
        let a = array![[1., 1.], [1., 2.], [0., 3.], [0., 4.]];

        let b = array![[0., 1.]];

        let c = array![[1., 2.], [1., 3.], [0., 4.], [0., 5.]];

        // We can add because the shapes are compatible even if not equal.
        // The `b` array is shape 1 × 2 but acts like a 4 × 2 array.
        assert!(c == a + b);
    }
}
