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
fn main() {
    // The Array<A, D> is parameterized by A for the element type and D for the dimensionality.
    // through turbofish syntax, and let it infer the dimensionality type
    let a: Array<f32, _> = zeros_with_dim((1, 3, 2));
    println!("{:?}", a);

    let a: Array<f64, _> = zeros_with_dim((1, 3, 2));
    println!("{:?}", a);

    let arr = Array::from_vec(vec![1, 2, 2, 3, 1])
        .into_shape_with_order(ndarray::IxDyn(&[5]))
        .unwrap();

    let map = array_value_counts(&arr);

    println!("{:?}", map);

    let map_unique_key = map.keys();
    println!("{:?}", map_unique_key);

    let len_map = map.len();
    println!("{:?}", len_map);

    //init empty crosstab
    let zero_crosstab = Array::<f64, _>::zeros((len_map, len_map));
    println!("{:?}", zero_crosstab);
    let b = (1, 3, 2);
    print_type(&b);
}
