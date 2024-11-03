use ndarray::prelude::*;
use ndarray::Array;
fn main() {
    // The Array<A, D> is parameterized by A for the element type and D for the dimensionality.
    // through turbofish syntax, and let it infer the dimensionality type
    let a = Array::<f64, _>::zeros((1, 3, 2).f());
    println!("{:?}", a);
    //dimensionality explicitly Array::<f64, Ix3>::zeros(...), withIx3
    let a = Array::<f64, Ix3>::zeros((1, 3, 2).f());
    println!("{:?}", a);

    //f()) makes a column major, so without f() it will be row major
    let a = Array::<f64, Ix3>::zeros((1, 3, 2));
    println!("{:?}", a);
}
