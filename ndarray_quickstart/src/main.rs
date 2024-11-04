use ndarray::Array;
use ndarray::Dimension;
use ndarray::Shape;
use ndarray::ShapeBuilder;

//the zero with dim is generic over some type Sh
fn zeros_with_dim<Sh>(dim: Sh) -> Array<f64, <Sh as ShapeBuilder>::Dim>
where
    Sh: ShapeBuilder<Dim: Dimension>,
{
    // # original codes examples
    // let a = Array::<f64, _>::zeros((1, 3, 2).f());

    // # dimensionality explicitly Array::<f64, Ix3>::zeros(...), withIx3
    // let a = Array::<f64, Ix3>::zeros((1, 3, 2).f());

    // # f()) makes a column major, so without f() it will be row major
    // let a = Array::<f64, Ix3>::zeros((1, 3, 2));
    Array::<f64, _>::zeros(dim)
}

fn print_type<T>(_: &T) {
    println!("{:?}", std::any::type_name::<T>());
}
fn main() {
    // The Array<A, D> is parameterized by A for the element type and D for the dimensionality.
    // through turbofish syntax, and let it infer the dimensionality type
    let a = zeros_with_dim((1, 3, 2));
    println!("{:?}", a);

    let b = (1, 3, 2);
    print_type(&b);
}
