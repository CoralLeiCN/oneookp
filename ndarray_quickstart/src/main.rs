use ndarray::Array;
use ndarray::Dimension;
// use ndarray::Shape;
use ndarray::ShapeBuilder;
use num_traits::Zero;

// trait Real {}

// impl Real for f32 {}
// impl Real for f64 {}

//the zero with dim is generic over some type Sh
fn zeros_with_dim<T, Sh>(dim: Sh) -> Array<T, <Sh as ShapeBuilder>::Dim>
where
    T: Zero + Clone,
    Sh: ShapeBuilder<Dim: Dimension>,
{
    Array::<T, _>::zeros(dim)
}
// # original codes examples
// let a = Array::<f64, _>::zeros((1, 3, 2).f());

// # dimensionality explicitly Array::<f64, Ix3>::zeros(...), withIx3
// let a = Array::<f64, Ix3>::zeros((1, 3, 2).f());

// # f() makes a column major, so without f() it will be row major
// let a = Array::<f64, Ix3>::zeros((1, 3, 2));

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
    let b = (1, 3, 2);
    print_type(&b);
}
